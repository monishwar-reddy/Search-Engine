import os
import subprocess

def main():
    html_path = os.path.abspath("report/report.html")
    pdf_path = os.path.abspath("report/report.pdf")
    
    print(f"Generating PDF from: {html_path}")
    print(f"Destination PDF path: {pdf_path}")
    
    # Common paths for Microsoft Edge on Windows
    edge_paths = [
        "msedge",  # If registered in PATH
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        # Check Chrome as fallback
        "chrome",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
    ]
    
    success = False
    for path in edge_paths:
        try:
            print(f"Trying to render PDF using: {path}")
            cmd = [
                path,
                "--headless",
                f"--print-to-pdf={pdf_path}",
                html_path
            ]
            # Run the command
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=15)
            if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
                print(f"Successfully generated PDF using {path}!")
                success = True
                break
        except Exception as e:
            # Try next path
            continue
            
    if not success:
        print("Error: Could not automatically generate PDF. Please ensure Microsoft Edge or Chrome is installed.")
        print(f"You can manually print {html_path} to PDF via your web browser.")
    else:
        print("PDF report generation complete!")

if __name__ == "__main__":
    main()
