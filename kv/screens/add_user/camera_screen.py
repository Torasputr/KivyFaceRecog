from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.image import Image
from utils.camera import gen_frames, release_camera, take_pictures
from utils.model import prepare_data, train_deep_learning_classifier, save_model, save_label_encoder
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import threading

Builder.load_file('kv/screens/add_user/camera_screen.kv')

class AddUserCameraScreen(Screen):
    user = StringProperty("")
    user_id = StringProperty("")
    frame_generator = None

    def on_enter(self, *args):
        self.frame_generator = gen_frames()
        Clock.schedule_interval(self.update_frame, 1/30)

    def update_frame(self, dt):
        try:
            frame = next(self.frame_generator)
            if frame is not None:
                texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
                texture.blit_buffer(frame.tobytes(), colorfmt='bgr', bufferfmt='ubyte')
                self.ids.camera_image.texture = texture
        except StopIteration:
            self.frame_generator = None

    def take_picture(self):
        print(f"Starting picture-taking process for user ID: {self.user_id}")
        
        # Run picture-taking in a background thread and trigger training once done
        def picture_callback():
            take_pictures(self.user_id)
            print("Finished taking pictures, starting model training...")
            self.start_model_training()

        # Start the picture-taking process in a background thread
        threading.Thread(target=picture_callback, daemon=True).start()

    def navigate_to_train_screen(self, user_id, user):
        train_screen = self.manager.get_screen("add_user_train")
        train_screen.user_id = user_id
        train_screen.user = user
        self.manager.current = "add_user_train"
        
    # def start_model_training(self):
    #     # This function will be called after picture-taking completes
    #     def model_training_callback():
    #         try:
    #             print("Preparing Data for Model Training")
    #             embeddings, labels = prepare_data()
    #             print("Training CNN classifier")
    #             classifier, label_encoder = train_deep_learning_classifier(embeddings, labels)
    #             print("Saving the trained model")
    #             save_model(classifier)
    #             save_label_encoder(label_encoder)
    #             print("Model training and saving completed successfully!")
    #         except Exception as e:
    #             print(f"Error during model training: {e}")

    #     # Start the model training process in a new background thread
    #     threading.Thread(target=model_training_callback, daemon=True).start()

    def on_leave(self, *args):
        release_camera()
        Clock.unschedule(self.update_frame)