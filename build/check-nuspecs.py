"""Usage: python build/check-nuspecs.py <directory of the packages>. Each package has the URL of the repository and of the project, so
that NuGet links its source; and the README, the icon, the third-party notices, and a lib folder for each target."""
import glob, os, re, sys, zipfile

expected = 'https://github.com/ChromaDotNet/ChromaDB.Embeddings'
files = ['README.md', 'icon.png', 'THIRD-PARTY-NOTICES.md', 'lib/net8.0/ChromaDotNet.Embeddings.dll',
         'lib/net462/ChromaDotNet.Embeddings.dll', 'buildTransitive/net462/ChromaDotNet.Embeddings.targets']
failures = 0
packages = sorted(glob.glob(os.path.join(sys.argv[1], '*.nupkg')))
for path in packages:
    with zipfile.ZipFile(path) as package:
        names = package.namelist()
        nuspec = package.read(next(n for n in names if n.endswith('.nuspec'))).decode('utf-8')
    repository = re.search(r'<repository [^>]*url="([^"]*)"', nuspec)
    project = re.search(r'<projectUrl>(.*?)</projectUrl>', nuspec)
    missing = [f for f in files if f not in names]
    groups = re.findall(r'<group targetFramework="([^"]*)"', nuspec)
    ok = (repository is not None and repository.group(1) == expected and project is not None and project.group(1) == expected
          and not missing and sorted(groups) == ['.NETFramework4.6.2', 'net8.0'])
    print(('ok   ' if ok else 'FAIL ') + os.path.basename(path) + (f' (missing {", ".join(missing)})' if missing else '')
          + f' dependency groups {", ".join(groups)}')
    failures += 0 if ok else 1
if not packages:
    print('FAIL no packages')
    failures = 1
sys.exit(failures)
