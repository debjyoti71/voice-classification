import pandas as pd

from VoiceClassification.new_voice_input import VoiceSampleCollector
from VoiceClassification.train import VoiceTrainer
from VoiceClassification.predict_voice_input import VoiceAuthenticator

def new_entry():
    name_input = input("Enter name: ")
    collector = VoiceSampleCollector(name_input)
    collector.run()
    
    df = pd.read_csv('data/audio_features_with_errors.csv')  # Load after data is added
    Label_to_train = name_input
    trainer = VoiceTrainer(df)
    trainer.run(Label_to_train)

def predict_voice():
    va = VoiceAuthenticator()
    deb_label_to_predict = input("Input your name: ")
    va.process_new_input(deb_label_to_predict)

if __name__ == "__main__":
    choice = input("Choose an option:\n1. New Entry\n2. Predict Voice\nEnter 1 or 2: ")
    if choice == '1':
        new_entry()
    elif choice == '2':
        predict_voice()
    else:
        print("Invalid choice. Please enter 1 or 2.")
