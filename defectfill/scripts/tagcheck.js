const fs = require('fs');
let s = fs.readFileSync(process.argv[2], 'utf8');
s = s.replace(/<!--[\s\S]*?-->/g, '');
s = s.replace(/\s(?:placeholder|value|src|id|class|name|href|title|data-[a-z]+)="[^"]*"/gi, '');
s = s.replace(/\s(?:placeholder|value|src|id|class|name|href|title|data-[a-z]+)='[^']*'/gi, '');
const voidTags = new Set(['input', 'img', 'br', 'hr', 'meta', 'link']);
const stack = [];
const re = /<\/?([a-zA-Z][a-zA-Z0-9]*)((?:\s[^<>]*)?)\/?>/g;
let m;
let errors = [];
let line = 1;
let idx = 0;
while ((m = re.exec(s))) {
  const before = s.slice(idx, m.index).split('\n').length - 1;
  line += before;
  idx = m.index;
  const full = m[0];
  const tag = m[1];
  const isClose = /^<\//.test(full);
  const isSelfClose = /\/>$/.test(full);
  if (!isClose && !isSelfClose && !voidTags.has(tag)) {
    stack.push({ tag, line });
  } else if (isClose) {
    const top = stack.pop();
    if (!top) errors.push('Unmatched close </' + tag + '> at line ' + line);
    else if (top.tag !== tag) errors.push('Mismatch: open <' + top.tag + '> line ' + top.line + ' closed by </' + tag + '> line ' + line);
  }
  line += full.split('\n').length - 1;
}
if (stack.length) errors.push('Unclosed: ' + stack.map(x => '<' + x.tag + '> line ' + x.line).join(', '));
if (errors.length) console.log('ERRORS:\n' + errors.join('\n'));
else console.log('OK: all tags balanced');
