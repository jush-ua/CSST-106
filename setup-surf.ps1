# Prerequisite: Visual Studio 2022 C++ build tools and Windows SDK.
# This builds OpenCV only inside an isolated environment and project build folder.
param([string]$BuildRoot = '')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.surf-venv\Scripts\python.exe')) { py -3.11 -m venv .surf-venv }
$surfPython = (Resolve-Path '.surf-venv\Scripts\python.exe').Path
& $surfPython -m pip install --no-compile 'numpy<2' 'cmake>=3.27,<4' ninja
if ($LASTEXITCODE -ne 0) { throw 'SURF build dependencies failed to install.' }
$sourceRoot = if ($BuildRoot) { [System.IO.Path]::GetFullPath($BuildRoot) } else { Join-Path $PSScriptRoot 'data\surf-build' }
New-Item -ItemType Directory -Force $sourceRoot | Out-Null
foreach ($repo in @('opencv','opencv_contrib')) {
    $target = Join-Path $sourceRoot $repo
    if (-not (Test-Path $target)) { git clone --depth 1 --branch 4.12.0 "https://github.com/opencv/$repo.git" $target }
    if ($LASTEXITCODE -ne 0) { throw "Could not obtain $repo sources." }
}
$purelib = & $surfPython -c 'import sysconfig; print(sysconfig.get_path("purelib"))'
$numpyIncludes = & $surfPython -c 'import numpy; print(numpy.get_include())'
$pythonIncludes = & $surfPython -c 'import sysconfig; print(sysconfig.get_path("include"))'
$pythonLibrary = & $surfPython -c 'import sys, os; print(os.path.join(sys.base_prefix, "libs", "python311.lib"))'
# OpenCV expands these paths inside CMake strings; backslashes become escapes.
$surfPython = $surfPython.Replace('\','/')
$purelib = $purelib.Replace('\','/')
$numpyIncludes = $numpyIncludes.Replace('\','/')
$pythonIncludes = $pythonIncludes.Replace('\','/')
$pythonLibrary = $pythonLibrary.Replace('\','/')
$cmake = Join-Path $PSScriptRoot '.surf-venv\Scripts\cmake.exe'
& $cmake -S "$sourceRoot\opencv" -B "$sourceRoot\build" -G 'Visual Studio 17 2022' -A x64 `
    '-DOPENCV_ENABLE_NONFREE=ON' "-DOPENCV_EXTRA_MODULES_PATH=$sourceRoot\opencv_contrib\modules" `
    '-DBUILD_LIST=core,imgproc,imgcodecs,features2d,flann,xfeatures2d,python3' `
    '-DBUILD_TESTS=OFF' '-DBUILD_PERF_TESTS=OFF' '-DBUILD_EXAMPLES=OFF' `
    '-DBUILD_SHARED_LIBS=OFF' '-DWITH_OPENCL=OFF' '-DWITH_FFMPEG=OFF' '-DWITH_IPP=OFF' `
    '-DWITH_JPEG=OFF' '-DWITH_TIFF=OFF' '-DWITH_WEBP=OFF' '-DWITH_OPENEXR=OFF' '-DWITH_OPENJPEG=OFF' `
    '-DWITH_JASPER=OFF' '-DWITH_AVIF=OFF' '-DWITH_ITT=OFF' '-DWITH_LAPACK=OFF' `
    '-DCPU_BASELINE=SSE2' '-DCPU_DISPATCH=' `
    '-DBUILD_JAVA=OFF' '-DBUILD_opencv_apps=OFF' `
    "-DPYTHON3_EXECUTABLE=$surfPython" "-DPYTHON3_PACKAGES_PATH=$purelib" `
    "-DOPENCV_PYTHON3_INSTALL_PATH=$purelib" "-DPYTHON3_NUMPY_INCLUDE_DIRS=$numpyIncludes" "-DPYTHON3_INCLUDE_DIR=$pythonIncludes" `
    "-DPYTHON3_LIBRARY=$pythonLibrary" "-DPYTHON3_LIBRARIES:INTERNAL=$pythonLibrary" `
    '-DBUILD_opencv_python3=ON'
if ($LASTEXITCODE -ne 0) { throw 'SURF CMake configuration failed. Install VS 2022 C++ Build Tools/SDK and retry.' }
& $cmake --build "$sourceRoot\build" --config Release --parallel 4
if ($LASTEXITCODE -ne 0) { throw 'SURF compilation failed.' }
& $cmake --install "$sourceRoot\build" --config Release --prefix "$sourceRoot\install"
if ($LASTEXITCODE -ne 0) { throw 'SURF installation failed.' }
& $surfPython -c 'import cv2; print(cv2.__version__); cv2.xfeatures2d.SURF_create(400); print("SURF available")'
if ($LASTEXITCODE -ne 0) { throw 'SURF import/capability verification failed; inspect the build logs.' }
