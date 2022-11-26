#!/bin/zsh
python -m pip install -r requirement.txt
unzip ./save_models/test.pth.zip
mv test.pth save_models
/usr/libexec/PlistBuddy -c 'Add :CFBundleIdentifier string "rumps"' /Users/zhuangcheng/opt/anaconda3/bin/Info.plist 
rm -rf build dist
python setup.py py2app -A
open ./dist/Im2Latex.app