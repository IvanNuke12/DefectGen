const fs = require('fs');
const js = fs.readFileSync('frontend/app.js', 'utf8');
const html = fs.readFileSync('frontend/index.html', 'utf8');
const ids = [...new Set([...js.matchAll(/\$\("#([A-Za-z0-9_]+)"\)/g)].map((m) => m[1]))].sort();
const missing = ids.filter((id) => !html.includes('id="' + id + '"'));
console.log('IDs usados por app.js: ' + ids.length);
console.log('Faltan en index.html: ' + (missing.length ? missing.join(', ') : 'ninguno'));
