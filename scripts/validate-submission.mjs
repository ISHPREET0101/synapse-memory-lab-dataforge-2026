import { readFile, stat } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const failures = [];

const mandatoryFiles = [
  '.env.example',
  'ARCHITECTURE.md',
  'LICENSE',
  'README.md',
  'docs/BLOG.md',
  'docs/CONCEPT_SUMMARY.md',
  'docs/JUDGE_DEFENSE.md',
  'docs/PAPERS.md',
  'docs/SOURCE_AND_LICENSES.md',
  'index.html',
  'output/pdf/Synapse_Memory_Lab_Blog.pdf',
  'output/pdf/Synapse_Memory_Lab_Concept_Summary.pdf',
  'package-lock.json',
  'package.json',
  'src/main.ts',
  'tests/engine.test.ts',
  'tsconfig.json',
  'vite.config.ts'
];

async function readRequired(relativePath) {
  const absolutePath = path.join(root, relativePath);
  try {
    const metadata = await stat(absolutePath);
    if (!metadata.isFile() || metadata.size === 0) {
      failures.push(`${relativePath}: must be a non-empty file`);
      return null;
    }
    return await readFile(absolutePath, 'utf8');
  } catch (error) {
    if (error?.code === 'ENOENT') {
      failures.push(`${relativePath}: missing mandatory file`);
      return null;
    }
    failures.push(`${relativePath}: could not be read (${error.message})`);
    return null;
  }
}

function parseJson(contents, relativePath) {
  if (contents === null) return null;
  try {
    return JSON.parse(contents);
  } catch (error) {
    failures.push(`${relativePath}: invalid JSON (${error.message})`);
    return null;
  }
}

const fileContents = new Map(
  await Promise.all(mandatoryFiles.map(async (file) => [file, await readRequired(file)]))
);

const packageJson = parseJson(fileContents.get('package.json'), 'package.json');
const packageLock = parseJson(fileContents.get('package-lock.json'), 'package-lock.json');

if (packageJson) {
  const requiredScripts = ['typecheck', 'test', 'build', 'preview', 'validate:submission', 'check'];
  for (const script of requiredScripts) {
    if (typeof packageJson.scripts?.[script] !== 'string' || packageJson.scripts[script].trim() === '') {
      failures.push(`package.json: missing non-empty \"${script}\" script`);
    }
  }
  if (packageJson.private !== true) {
    failures.push('package.json: \"private\" must be true to prevent accidental publication');
  }
}

if (packageJson && packageLock) {
  const lockRoot = packageLock.packages?.[''];
  if (!lockRoot) {
    failures.push('package-lock.json: missing root package entry');
  } else {
    for (const field of ['name', 'version']) {
      if (lockRoot[field] !== packageJson[field]) {
        failures.push(`package-lock.json: root ${field} does not match package.json`);
      }
    }
    for (const [name, version] of Object.entries(packageJson.devDependencies ?? {})) {
      if (lockRoot.devDependencies?.[name] !== version) {
        failures.push(`package-lock.json: devDependency ${name} is not synchronized`);
      }
    }
  }
}

const viteConfig = fileContents.get('vite.config.ts');
if (viteConfig !== null) {
  if (!/base\s*:\s*['"]\.\/['"]/.test(viteConfig)) {
    failures.push("vite.config.ts: base must be './' for relative static deployment");
  }
  if (!/emptyOutDir\s*:\s*true/.test(viteConfig)) {
    failures.push('vite.config.ts: emptyOutDir must be true for clean builds');
  }
}

const distIndex = await readRequired('dist/index.html');
if (distIndex !== null) {
  const assetReferences = [...distIndex.matchAll(/(?:src|href)=["']([^"']+)["']/g)].map((match) => match[1]);
  const localAssets = assetReferences.filter((reference) => !/^(?:data:|https?:|#)/.test(reference));

  for (const reference of localAssets) {
    if (reference.startsWith('/')) {
      failures.push(`dist/index.html: root-relative asset is not portable (${reference})`);
      continue;
    }

    const cleanReference = reference.split(/[?#]/, 1)[0];
    const assetPath = path.resolve(root, 'dist', cleanReference);
    const distRoot = `${path.resolve(root, 'dist')}${path.sep}`;
    if (!assetPath.startsWith(distRoot)) {
      failures.push(`dist/index.html: asset escapes dist directory (${reference})`);
      continue;
    }

    try {
      const metadata = await stat(assetPath);
      if (!metadata.isFile() || metadata.size === 0) {
        failures.push(`dist/index.html: asset is missing or empty (${reference})`);
      }
    } catch {
      failures.push(`dist/index.html: referenced asset is missing (${reference})`);
    }
  }
}

if (failures.length > 0) {
  console.error(`Submission validation failed with ${failures.length} issue(s):`);
  for (const failure of failures) console.error(`  - ${failure}`);
  process.exitCode = 1;
} else {
  console.log(`Submission validation passed (${mandatoryFiles.length + 1} mandatory files checked).`);
  console.log('Static build uses relative asset paths and every referenced local asset exists.');
}
