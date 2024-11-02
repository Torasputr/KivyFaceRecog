from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from utils.user_manager import save_user_to_csv, create_user_directory

Builder.load_file("kv/screens/add_user/index_screen.kv")

class AddUserIndexScreen(Screen):
    name_input = ObjectProperty(None)
    def register_user(self):
        name = self.name_input.text.strip()
        if name:
            user_id = save_user_to_csv(name)
            create_user_directory(user_id)
            print(f"User {name} registered with ID {user_id}")
            camera_screen = self.manager.get_screen("add_user_camera")
            camera_screen.username = name
            camera_screen.user_id = user_id
            self.manager.current = "add_user_camera"
            
        else:
            print("Please enter a valid name")
