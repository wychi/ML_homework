MacOSX

clang: warning: using sysroot for 'iPhoneOS' but targeting 'MacOSX'

Check environment variables first.
```
export | grep sdk

SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS10.2.sdk
SDK_DIR=/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS10.2.sdk
SDK_DIR_iphoneos10_2=/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS10.2.sdk
```

Specify the correct sdk before calling pip install
```
SDKROOT=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.12.sdk pip install --user numpy scipy matplotlib ipython jupyter pandas sympy nose
```
