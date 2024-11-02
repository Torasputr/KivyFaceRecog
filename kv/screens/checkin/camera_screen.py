from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.image import Image
from utils.camera import gen_frames, release_camera, take_auth_pic
from utils.model import authenticate_user_with_cnn, load_keras_model
from kivy.clock import Clock
from kivy.graphics.texture import Texture

Builder.load_file('kv/screens/checkin/camera_screen.kv')

class CheckinCameraScreen(Screen):
    frame_generator = None

    def on_enter(self, *args):
        self.frame_generator = gen_frames()
        Clock.schedule_interval(self.update_frame, 1/24)

    def update_frame(self, dt):
        try:
            frame = next(self.frame_generator)
            if frame is not None:
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
                self.ids.camera_image.texture = texture
        except StopIteration:
            self.frame_generator = None

    def start_recog(self):
        result = authenticate_user_with_cnn()  # Ensure this function does not use Flask features
        if result:  # Check if authentication was successful
            # Handle successful authentication (e.g., navigate to a success screen)
            self.manager.current = "home"  # or any success screen
        else:
            # Handle failure (e.g., show a message)
            print("Authentication failed.")

    def on_leave(self, *args):
        release_camera()
        Clock.unschedule(self.update_frame)
