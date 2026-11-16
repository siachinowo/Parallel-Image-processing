# Contributing Guide

Welcome! Please follow these steps so we don't break each other's work.

## Golden rules
- Never push directly to `main`. It's protected.
- One feature or fix per branch.
- Pull the latest `main` before you start new work.

## Step-by-step

1. **Get the latest code**
   git checkout main
   git pull

2. **Create your own branch** (use your name and a short description)
   git checkout -b yourname/short-description

3. **Do your work**, then save it:
   git add .
   git commit -m "Short description of what you changed"

4. **Push your branch**
   git push -u origin yourname/short-description

5. **Open a Pull Request** on GitHub (a yellow banner will appear with a "Compare & pull request" button). Fill in the template.

6. **Wait for review.** The project owner will approve and merge it, or ask for changes. If changes are requested, just commit and push to the same branch; the PR updates automatically.

## Tips
- Commit small and often.
- Write clear commit messages ("Add login form", not "stuff").
- If you get a merge conflict or feel stuck, ask for help before forcing anything.
- Don't commit large video files or secrets (passwords, API keys).
