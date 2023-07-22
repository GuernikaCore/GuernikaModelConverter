# Guernika Model Converter

This repository contains a model converter compatible with [Guernika](https://apps.apple.com/app/id1660407508).

## <a name="converting-models-to-guernika"></a> Converting Models to Guernika

**WARNING:** Xcode is required to convert models:

 - Make sure you have [Xcode](https://apps.apple.com/app/id497799835) installed.
 
 - Once installed run the following commands:

```shell
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer/
sudo xcodebuild -license accept
```

 - You should now be ready to start converting models!

**Step 1:** Download and install [`Guernika Model Converter`](https://huggingface.co/Guernika/CoreMLStableDiffusion/resolve/main/GuernikaModelConverter.dmg).

[<img alt="Guernika Model Converter icon" src="https://huggingface.co/Guernika/CoreMLStableDiffusion/resolve/main/GuernikaModelConverter_AppIcon.png" width="256pt" />](https://huggingface.co/Guernika/CoreMLStableDiffusion/resolve/main/GuernikaModelConverter.dmg)

**Step 2:** Launch `Guernika Model Converter` from your `Applications` folder, this app may take a few seconds to load.

**Step 3:** Once the app has loaded you will be able to select what model you want to convert:

  - You can input the model identifier (e.g. CompVis/stable-diffusion-v1-4) to download from Hugging Face. You may have to log in to or register for your [Hugging Face account](https://huggingface.co), generate a [User Access Token](https://huggingface.co/settings/tokens) and use this token to set up Hugging Face API access by running `huggingface-cli login` in a Terminal window.
    
  - You can select a local model from your machine: `Select local model`

  - You can select a local .CKPT model from your machine: `Select CKPT`

<img alt="Guernika Model Converter interface" src="https://huggingface.co/Guernika/CoreMLStableDiffusion/resolve/main/GuernikaModelConverter_screenshot.png" />

**Step 4:** Once you've chosen the model you want to convert you can choose what modules to convert and/or if you want to chunk the UNet module (recommended for iOS/iPadOS devices).

**Step 5:** Once you're happy with your selection click `Convert to Guernika` and wait for the app to complete conversion.
**WARNING:** This command may download several GB worth of PyTorch checkpoints from Hugging Face and may take a long time to complete (15-20 minutes on an M1 machine).
