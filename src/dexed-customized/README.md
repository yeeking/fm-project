# Various hacks of dexed 

## Hacked version of dexed for rendering patches

This one renders a dataset of WAV files by loading MIDI cartridge files into dexed and playing notesd on every patch.

How to build my hacky version of dexed that just renders a folder of carts to WAVs:

First build it:

```
git clone https://github.com/asb2m10/dexed.git
# to render carts to wavs
cp dexed-carts-to-wavs/* ./dexed/Source/
# OR to render carts to parameter txt files
cp dexed-parameters-to-files/* ./dexed/Source/
cd dexed
git submodule update --init --recursive
cmake -B build .
cmake --build build --config Release -j 12 # j is number of threads for build
```

Then run it:
```
/build/Source/Dexed_artefacts/Standalone/Dexed 
```


## Macos build: I updated the macos target to allow me to have updated STD:: I think... add this to CMakeLists.txt somewhere

```
if (${CMAKE_SYSTEM_NAME} STREQUAL "iOS")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "Minimum OS X deployment version" FORCE)
endif()
````


