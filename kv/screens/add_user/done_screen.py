from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.clock import Clock
from kivy.properties import StringProperty

Builder.load_file('kv/screens/add_user/done_screen.kv')

class AddUserDoneScreen(Screen):
    username = StringProperty("")
    def on_enter(self, *args):
        Clock.schedule_once(self.home, 2)
    
    def home(self, dt):
        self.manager.current = "home"