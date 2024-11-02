from kivy.app import App
from kivy.lang import Builder

from kv.screens.home.home_screen import HomeScreen
from kv.screens.add_user.index_screen import AddUserIndexScreen
from kv.screens.add_user.camera_screen import AddUserCameraScreen
from kv.screens.add_user.train_screen import AddUserTrainScreen
from kv.screens.add_user.done_screen import AddUserDoneScreen
from kv.screens.checkin.camera_screen import CheckinCameraScreen

class FaceRecognitionApp(App):
    def build(self):
        sm = Builder.load_file("kv/main.kv")
        sm.current = "home"
        return sm
    
if __name__ == "__main__":
    FaceRecognitionApp().run()