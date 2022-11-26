### Before Installation

Put the model file `test.pth` under `save_models/` folder (omit this step to use our built-in model)

### CLI Application
#### Install Instruction
1. Install the dependency using the following command
```
python -m pip install -r requirement.txt
```
2. run the program from the command line
```
python api.py
```


### MacOS Application
#### Run directly using python

1. Install the dependency using the following command
```
python -m pip install -r requirement.txt
```
2. run following command in terminal to start the app
```
python im2latex.py
```

#### Install as an Mac application

1. Add execution permission for `./start.sh` using following command
```
chmod +x ./start.sh
```
2. Run `./start.sh`, it will automatically compile and run the app, you should able to see the app icon in menubar after install.

#### Remarks
* First run of the program will be ask you to grant notification and screen recording permission. Please first quit the app and enter the mac's setting, allow the notification and add the app to the screen recording permission list.
* You can move the `dist/Im2Latex.app` file to mac's `Application/` folder for quick access
* Before start to crop formula, try to use the `calibrate` function first to adjust the resize ratio for your workspace.
