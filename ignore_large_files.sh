#!/bin/bash
# This script finds files larger than 100MB and adds them to .gitignore

max_size=100M  # 100MB size limit

echo "🔍 Searching for files larger than $max_size..."
find . -type f -size +$max_size | while read -r file; do
    # Remove the leading './' for cleaner paths
    file=${file#./}
    # Check if the file is already in .gitignore
    if ! grep -qxF "$file" .gitignore; then
        echo "Adding $file to .gitignore"
        echo "$file" >> .gitignore
    fi
done

echo "✅ Finished updating .gitignore with large files."
