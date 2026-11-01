#!/usr/bin/env bash
# Build the guide: Markdown chapters -> one HTML file (print to PDF from the browser).
# Run from docs/guide:  ./build.sh
set -e
cd "$(dirname "$0")"
mkdir -p build
cat > build/style.css <<'CSS'
body { font-family: "Segoe UI", Tahoma, Arial, sans-serif; max-width: 52rem; margin: 2rem auto; padding: 0 1rem; line-height: 1.65; }
code, pre { font-family: Consolas, "DejaVu Sans Mono", monospace; font-size: 0.9em; }
pre { background: #f5f5f5; padding: 0.8rem; overflow-x: auto; white-space: pre-wrap; }
table { border-collapse: collapse; margin: 1rem 0; } td, th { border: 1px solid #999; padding: 0.3rem 0.6rem; }
h1 { page-break-before: always; } h1:first-of-type { page-break-before: avoid; }
CSS
pandoc [0-9][0-9]-*.md A-*.md B-*.md C-*.md -o build/guide.html --standalone --embed-resources \
  --toc --toc-depth=2 -c build/style.css --metadata title="Arabic Sentiment Benchmark: The Guide" 2>/dev/null \
  || pandoc [0-9][0-9]-*.md -o build/guide.html --standalone --embed-resources --toc --toc-depth=2 \
       -c build/style.css --metadata title="Arabic Sentiment Benchmark: The Guide"
echo "built build/guide.html"
