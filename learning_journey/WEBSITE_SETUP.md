# 🌐 Website Setup Guide

This learning journey is configured to be published as a beautiful documentation website using [Retype](https://retype.com/).

## Quick Start

### Option 1: Local Development

```bash
# Install Retype globally
npm install retypeapp --global

# Navigate to learning journey folder
cd learning_journey

# Build and serve locally
retype watch

# Open browser to http://localhost:5000
```

### Option 2: Build Static Site

```bash
# Build the site
retype build

# Output will be in .retype/ folder
# Open .retype/index.html in browser
```

### Option 3: GitHub Pages (Automated)

1. Push this repository to GitHub
2. Go to repository Settings → Pages
3. Select "Deploy from a branch"
4. Choose the "retype" branch and / (root) folder
5. Your site will be live at `https://yourusername.github.io/repo-name/`

The GitHub Actions workflow (`.github/workflows/retype.yml`) will automatically build and deploy on every push to main.

## Site Structure

```
learning_journey/
├── retype.yml          # Retype configuration
├── README.md           # Home page (this becomes the landing page)
├── week1_foundations/
│   └── README.md       # Week 1 content
├── week2_tokenization/
│   └── README.md       # Week 2 content
├── ...
└── .retype/            # Generated site (don't edit manually)
```

## Customization

Edit `retype.yml` to customize:
- Title and branding
- Colors and theme
- Navigation sidebar
- Links and footer

## Features

✅ **Beautiful Documentation** - Clean, modern design
✅ **Dark/Light Mode** - Automatic theme switching
✅ **Search** - Built-in search functionality
✅ **Navigation** - Automatic sidebar navigation
✅ **Code Highlighting** - Syntax highlighted code blocks
✅ **Mobile Responsive** - Works on all devices
✅ **Fast** - Optimized static site

## Deployment Options

### GitHub Pages (Free)
- Already configured with GitHub Actions
- Automatic deployment on every push

### Netlify
```bash
# Build locally
retype build

# Deploy .retype/ folder to Netlify
```

### Vercel
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --cwd .retype
```

### Custom Server
```bash
# Build
retype build

# Copy .retype/ contents to your web server
scp -r .retype/* user@server:/var/www/html/
```

## Troubleshooting

### Issue: Site not building
```bash
# Check retype.yml syntax
retype build --verbose
```

### Issue: Links not working
- Ensure all internal links use relative paths
- Use `.md` extension in links (Retype converts them)

### Issue: Images not showing
- Place images in same folder as markdown or in `static/`
- Use relative paths: `./image.png`

## Learn More

- [Retype Documentation](https://retype.com/guides/getting-started/)
- [Markdown Syntax](https://retype.com/guides/formatting/)
- [Configuration Options](https://retype.com/configuration/project/)

---

Happy publishing! 🚀
