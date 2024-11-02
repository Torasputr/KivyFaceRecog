from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
import threading
from utils.model import prepare_data, train_deep_learning_classifier, save_model, save_label_encoder
from kivy.properties import StringProperty
from kivy.clock import Clock

Builder.load_file('kv/screens/add_user/train_screen.kv')

class AddUserTrainScreen(Screen):
    user_id = StringProperty("")
    username = StringProperty("")  

    def on_enter(self, *args):
        self.register_user()

    def register_user(self):
            print("Preparing Data")
            embeddings, labels = prepare_data()
            print("Starting Model Training...")
            classifier, label_encoder = train_deep_learning_classifier(embeddings, labels)
            save_model(classifier)
            save_label_encoder(label_encoder)
            print("Training completed successfully!")
            print("Switching to AddUserDoneScreen")
            camera_screen = self.manager.get_screen("add_user_done")
            camera_screen.username = self.name
            camera_screen.user_id = self.user_id
            self.manager.current = "add_user_done"