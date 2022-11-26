import rumps
import api 
import subprocess
import time

rumps.debug_mode(True)

class Im2Latex(rumps.App):
    def __init__(self):
        super(Im2Latex, self).__init__("Im2Latex", icon="./assets/latex.svg")
        with open('./assets/config.txt') as f:
            self.scale = float(f.readline().strip())
        rumps.notification("Im2Latex", None, "Im2Latex is running in the background, please calibrate before use")
        
    # crop image
    @rumps.clicked("Crop Image", key = "F1")
    def convert_image(self, _):
        try:
            api.convert(self.scale)
        except Exception as e:
            rumps.notification("Error", None, str(e))
        rumps.notification("Convert Image Done", None, "Result copied to clipboard")


    # calibrate
    @rumps.clicked("Calibrate Scale", key = "G2")
    def calibrate(self, _):

        rumps.notification("Calibrate", None, "Start calibrating...")
        t1 = time.time()
        return_str, scale = api.calibrate()
        self.scale = scale
        with open("./assets/config.txt", "w") as f:
            f.write(str(scale))
        f.close()
        t2 = time.time()
        rumps.notification("Calibrate", None, "Calibrate Done, takes {} seconds, result copied".format(round(t2-t1, 2)))
        

    # change scale
    @rumps.clicked("Change Scale", key = "R2")
    def change_scale(self, _):
        rumps.notification("Change Scale", None, "Quit the editor to continue")
        subprocess.call(["open", "./assets/config.txt", "--wait"])
        with open('./assets/config.txt', 'r') as f:
            self.scale = float(f.readline().strip())

if __name__ == "__main__":
    Im2Latex().run()
    