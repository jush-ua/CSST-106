# Prerequisite: Visual Studio 2022 C++ build tools and Windows SDK.
# This builds OpenCV only inside an isolated environment and project build folder.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.surf-venv\Scripts\python.exe')) { py -3.11 -m venv .surf-venv }
$surfPython = (Resolve-Path '.surf-venv\Scripts\python.exe').Path
& $surfPython -m pip install 'numpy<2' cmake ninja
if ($LASTEXITCODE -ne 0) { throw 'SURF build dependencies failed to install.' }
$sourceRoot = Join-Path $PSScriptRoot 'data\surf-build'
New-Item -ItemType Directory -Force $sourceRoot | Out-Null
foreach ($repo in @('opencv','opencv_contrib')) {
    $target = Join-Path $sourceRoot $repo
    if (-not (Test-Path $target)) { git clone --depth 1 --branch 4.12.0 "https://github.com/opencv/$repo.git" $target }
    if ($LASTEXITCODE -ne 0) { throw "Could not obtain $repo sources." }
}
$purelib = & $surfPython -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
$cmake = Join-Path $PSScriptRoot '.surf-venv\Scripts\cmake.exe'
& $cmake -S "$sourceRoot\opencv" -B "$sourceRoot\build" -G 'Visual Studio 17 2022' -A x64 `
    '-DOPENCV_ENABLE_NONFREE=ON' "-DOPENCV_EXTRA_MODULES_PATH=$sourceRoot\opencv_contrib\modules" `
    '-DBUILD_LIST=core,imgproc,imgcodecs,features2d,flann,xfeatures2d,python3' `
    '-DBUILD_TESTS=OFF' '-DBUILD_PERF_TESTS=OFF' '-DBUILD_EXAMPLES=OFF' `
    "-DPYTHON3_EXECUTABLE=$surfPython" "-DPYTHON3_PACKAGES_PATH=$purelib" `
    '-DBUILD_opencv_python3=ON'
if ($LASTEXITCODE -ne 0) { throw 'SURF CMake configuration failed. Install VS 2022 C++ Build Tools/SDK and retry.' }
& $cmake --build "$sourceRoot\build" --config Release --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'SURF compilation failed.' }
& $cmake --install "$sourceRoot\build" --config Release --prefix "$sourceRoot\install"
if ($LASTEXITCODE -ne 0) { throw 'SURF installation failed.' }
& $surfPython -c 'import cv2; print(cv2.__version__); cv2.xfeatures2d.SURF_create(400); print("SURF available")'
if ($LASTEXITCODE -ne 0) { throw 'SURF import/capability verification failed; inspect the build logs.' }
