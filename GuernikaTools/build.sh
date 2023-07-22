#!/bin/bash

pyinstaller GuernikaTools.spec --clean -y
codesign -s - -i com.guiyec.GuernikaModelConverter.GuernikaTools -o runtime --entitlements GuernikaTools.entitlements -f "../GuernikaModelConverter/GuernikaTools"
