import os
import argparse
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models

class FashionRetriever:
    def __init__(self, storage_path="data/qdrant_storage"):
        self.client = QdrantClient(path=storage_path)
        self.collection_name = "fashion_collection"
        
        # Load CLIP
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        
        # Vocabulary
        self.colors = ["black", "white", "blue", "red", "yellow", "green", "grey", "brown", "pink", "purple", "orange", "beige"]
        self.upper_types = ["shirt", "t-shirt", "hoodie", "blazer", "raincoat", "jacket", "dress", "sweater"]
        self.lower_types = ["pants", "jeans", "shorts", "skirt"]
        self.environments = ["office", "urban street", "park", "home", "beach", "nature path", "formal hall"]
        self.styles = ["formal business attire", "casual weekend wear", "sporty activewear", "cozy loungewear", "trendy streetwear"]

    def parse_query(self, query):
        """
        Parses a natural language query into structured attribute targets using proximity matching.
        """
        query_lower = query.lower()
        
        # 1. Identify colors and garments in the query and find their indices
        colors_found = []
        for c in self.colors:
            # Match whole words to avoid partial matches (e.g., 'red' inside 'ordered')
            idx = query_lower.find(c)
            while idx != -1:
                # check boundaries
                is_start = (idx == 0 or not query_lower[idx-1].isalnum())
                is_end = (idx + len(c) == len(query_lower) or not query_lower[idx+len(c)].isalnum())
                if is_start and is_end:
                    colors_found.append((c, idx))
                    break
                idx = query_lower.find(c, idx + 1)
                
        all_garments = self.upper_types + self.lower_types + ["tie"]
        garments_found = []
        for g in all_garments:
            idx = query_lower.find(g)
            while idx != -1:
                is_start = (idx == 0 or not query_lower[idx-1].isalnum())
                is_end = (idx + len(g) == len(query_lower) or not query_lower[idx+len(g)].isalnum())
                if is_start and is_end:
                    garments_found.append((g, idx))
                    break
                idx = query_lower.find(g, idx + 1)
                
        # 2. Proximity assignment: Associate colors with the closest garment (typically color before garment)
        associations = {}
        for c, c_idx in colors_found:
            best_garment = None
            min_dist = 9999
            for g, g_idx in garments_found:
                dist = g_idx - c_idx
                # We prefer the color to be right before the garment (positive distance)
                if dist > 0 and dist < min_dist:
                    min_dist = dist
                    best_garment = g
                # If color is slightly after the garment (e.g. "shirt in blue"), we also accept it
                elif dist < 0 and abs(dist) < 15 and abs(dist) < min_dist:
                    min_dist = abs(dist)
                    best_garment = g
            if best_garment:
                associations[best_garment] = c

        # 3. Build structured targets
        targets = {
            "upper_type": None,
            "upper_color": None,
            "lower_type": None,
            "lower_color": None,
            "has_tie": None,
            "tie_color": None,
            "environment": None,
            "style": None
        }
        
        # Extract garments and their associated colors
        for g, _ in garments_found:
            color = associations.get(g)
            if g in self.upper_types:
                targets["upper_type"] = g
                if color:
                    targets["upper_color"] = color
            elif g in self.lower_types:
                targets["lower_type"] = g
                if color:
                    targets["lower_color"] = color
            elif g == "tie":
                targets["has_tie"] = True
                if color:
                    targets["tie_color"] = color

        # 4. Check for environment
        for env in self.environments:
            # Support simple sub-words like 'office', 'park', 'street'
            words = env.split()
            if any(w in query_lower for w in words):
                targets["environment"] = env
                break
        # Special environment mapping rules
        if "street" in query_lower or "walk" in query_lower:
            targets["environment"] = "urban street"
        elif "office" in query_lower or "work" in query_lower:
            targets["environment"] = "office"
        elif "formal" in query_lower:
            targets["environment"] = "formal hall"
            targets["style"] = "formal business attire"

        # 5. Check for style
        if "formal" in query_lower or "business" in query_lower or "suit" in query_lower:
            targets["style"] = "formal business attire"
        elif "casual" in query_lower or "weekend" in query_lower:
            targets["style"] = "casual weekend wear"
        elif "sport" in query_lower or "active" in query_lower:
            targets["style"] = "sporty activewear"
        elif "cozy" in query_lower or "lounge" in query_lower:
            targets["style"] = "cozy loungewear"
        elif "streetwear" in query_lower:
            targets["style"] = "trendy streetwear"

        return targets

    def search(self, query, top_k=5):
        # 1. Parse structured targets from query
        targets = self.parse_query(query)
        
        # 2. Compute text features for the query using CLIP
        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            query_features = self.model.get_text_features(**inputs)
            if hasattr(query_features, "pooler_output"):
                query_features = query_features.pooler_output
        query_features = query_features / query_features.norm(dim=-1, keepdim=True)
        query_vector = query_features.squeeze(0).cpu().numpy().tolist()
        
        # 3. Retrieve candidates from Qdrant using vector search
        # Retrieve 50 candidates to allow re-ranking
        query_response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=50
        )
        search_results = query_response.points
        
        # 4. Compute hybrid scores
        ranked_results = []
        for res in search_results:
            payload = res.payload
            clip_score = res.score
            
            # Compute boost score
            boost = 0.0
            breakdown = {}
            
            # Check upper garment type and color matches
            if targets["upper_type"]:
                if payload["upper_type"] == targets["upper_type"]:
                    boost += 0.10
                    breakdown["upper_type"] = "match (+0.10)"
                    if targets["upper_color"] and payload["upper_color"] == targets["upper_color"]:
                        boost += 0.10
                        breakdown["upper_color"] = "match (+0.10)"
                elif targets["upper_color"] and payload["upper_color"] == targets["upper_color"]:
                    # General color match boost even if type doesn't match
                    boost += 0.05
                    breakdown["upper_color"] = "partial match (+0.05)"
            elif targets["upper_color"]:
                if payload["upper_color"] == targets["upper_color"]:
                    boost += 0.08
                    breakdown["upper_color"] = "match (+0.08)"
                    
            # Check lower garment type and color matches
            if targets["lower_type"]:
                if payload["lower_type"] == targets["lower_type"]:
                    boost += 0.10
                    breakdown["lower_type"] = "match (+0.10)"
                    if targets["lower_color"] and payload["lower_color"] == targets["lower_color"]:
                        boost += 0.10
                        breakdown["lower_color"] = "match (+0.10)"
                elif targets["lower_color"] and payload["lower_color"] == targets["lower_color"]:
                    boost += 0.05
                    breakdown["lower_color"] = "partial match (+0.05)"
            elif targets["lower_color"]:
                if payload["lower_color"] == targets["lower_color"]:
                    boost += 0.08
                    breakdown["lower_color"] = "match (+0.08)"
                    
            # Check tie matches
            if targets["has_tie"] is not None:
                if payload["has_tie"] == targets["has_tie"]:
                    boost += 0.08
                    breakdown["tie_status"] = "match (+0.08)"
                    if targets["tie_color"] and payload["tie_color"] == targets["tie_color"]:
                        boost += 0.12
                        breakdown["tie_color"] = "match (+0.12)"
                elif payload["has_tie"] and not targets["has_tie"]:
                    # Demote if person has a tie but user asked for no tie
                    boost -= 0.05
                    breakdown["tie_status"] = "mismatch (-0.05)"
            
            # Check environment matches
            if targets["environment"]:
                if payload["environment"] == targets["environment"]:
                    boost += 0.08
                    breakdown["environment"] = "match (+0.08)"
                    
            # Check style matches
            if targets["style"]:
                if payload["style"] == targets["style"]:
                    boost += 0.06
                    breakdown["style"] = "match (+0.06)"
                    
            hybrid_score = clip_score + boost
            ranked_results.append({
                "filename": payload["filename"],
                "path": payload["path"],
                "clip_score": float(clip_score),
                "boost_score": float(boost),
                "hybrid_score": float(hybrid_score),
                "payload": payload,
                "boost_breakdown": breakdown
            })
            
        # Re-sort candidates by hybrid score in descending order
        ranked_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        # Return top k
        return ranked_results[:top_k], targets

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Fashion Retriever")
    parser.add_argument("--query", type=str, required=True, help="Natural language query to search for")
    parser.add_argument("--k", type=int, default=5, help="Number of top results to return")
    args = parser.parse_args()
    
    retriever = FashionRetriever()
    results, targets = retriever.search(args.query, top_k=args.k)
    
    print("\n" + "="*50)
    print(f"Query: '{args.query}'")
    print(f"Parsed Targets: {targets}")
    print("="*50)
    
    for rank, res in enumerate(results, 1):
        print(f"\nRank {rank}: {res['filename']}")
        print(f"  Hybrid Score: {res['hybrid_score']:.4f} (CLIP: {res['clip_score']:.4f}, Boost: {res['boost_score']:.4f})")
        print(f"  Detected Payload:")
        print(f"    Upper: {res['payload']['upper_color']} {res['payload']['upper_type']}")
        print(f"    Lower: {res['payload']['lower_color']} {res['payload']['lower_type']}")
        print(f"    Tie: {'Yes' if res['payload']['has_tie'] else 'No'} (Color: {res['payload']['tie_color']})")
        print(f"    Environment: {res['payload']['environment']} | Style: {res['payload']['style']}")
        if res['boost_breakdown']:
            print(f"  Boost Breakdown: {res['boost_breakdown']}")
    print("="*50)
