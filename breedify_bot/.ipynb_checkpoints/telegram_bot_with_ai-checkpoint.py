import telebot
import json
import os
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
import random
import string
import time


from run_nn import predict_single_image

# Load configuration
with open(r'/home/jovyan/AI-dog-identifier-project/breedify_bot/config.json', 'r') as config_file:
    config = json.load(config_file)

BOT_TOKEN = config['BOT_TOKEN']
MODEL_PATH = config['MODEL_PATH']
FOLDER_PATH = config['FOLDER_PATH']
LABEL_PATH = config['LABEL_PATH']
TRAINING_FOLDER = config['SAVE_FOLDER_PATH']
USER_DATA = {}
RESULT_BREED = ''

bot = telebot.TeleBot(BOT_TOKEN)

def generate_code_for_image():
    return ''.join(random.choices(string.digits, k=10))

@bot.message_handler(commands=['start', 'restart'])
def start_command(message):
    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(
        InlineKeyboardButton("Find out the breed of the dog🐕", callback_data="try_breedify")
    )
    bot.send_message(
        message.chat.id,
        "Welcome to Breedify Bot! 🐶\n"
        "I'm here to help you identify the breed of your dog.\n"
        "Simply press the button to start.",
        reply_markup=markup
    )

@bot.message_handler(commands=['predict'])
def handle_predict_command(message):
    USER_DATA[message.chat.id] = {"text_command": "predict", "image_path": None, "predicted_breed": None}
    bot.reply_to(message, "Please send me an image of the dog!")

@bot.callback_query_handler(func=lambda call: call.data in ["try_breedify", "start_over"])
def handle_try_breedify(call):
    chat_id = call.message.chat.id
    try:
        bot.delete_message(chat_id, call.message.message_id)
    except telebot.apihelper.ApiTelegramException as e:
        if "message can't be deleted for everyone" in str(e):
            bot.answer_callback_query(call.id, "")

    USER_DATA[chat_id] = {"text_command": "predict", "image_path": None, "predicted_breed": None}
    bot.answer_callback_query(call.id)
    bot.send_message(chat_id, "Please send me an image of the dog!")

@bot.message_handler(content_types=['photo'])
def handle_image(message):
    if message.chat.id in USER_DATA and USER_DATA[message.chat.id].get("text_command") == "predict":
        sent_message = bot.send_message(message.chat.id, "Please wait, I am thinking...")

        image_path = os.path.join(FOLDER_PATH, f"image_{generate_code_for_image()}.jpg")

        try:
            file_info = bot.get_file(message.photo[-1].file_id)
            downloaded_file = bot.download_file(file_info.file_path)
            with open(image_path, "wb") as file:
                file.write(downloaded_file)
            USER_DATA[message.chat.id]["image_path"] = image_path
        except Exception as e:
            try:
                bot.delete_message(chat_id=message.chat.id, message_id=sent_message.message_id)
            except telebot.apihelper.ApiTelegramException:
                pass
            bot.reply_to(message, f"Failed to save the image: {e}")
            return

        # Process the image
        try:
            RESULT_BREED = predict_single_image(image_path, MODEL_PATH, LABEL_PATH)
            
            if "Error" in RESULT_BREED:
                try:
                    bot.delete_message(chat_id=message.chat.id, message_id=sent_message.message_id)
                except telebot.apihelper.ApiTelegramException:
                    pass
                bot.reply_to(message, f"An error occurred: {RESULT_BREED['Error']}")
            else:
                predicted_breed = RESULT_BREED["Predicted Breed"]
                confidence = RESULT_BREED["Confidence"]
                USER_DATA[message.chat.id]["predicted_breed"] = predicted_breed
                
                if confidence>0.7:
                    try:
                        bot.delete_message(chat_id=message.chat.id, message_id=sent_message.message_id)
                    except telebot.apihelper.ApiTelegramException:
                        pass
                    bot.reply_to(
                        message,
                        f"I am pretty sure that is: {predicted_breed.replace('_', ' ')}"
                    )

                    # Show feedback options
                    markup = InlineKeyboardMarkup()
                    markup.row_width = 1
                    markup.add(
                        InlineKeyboardButton("Yes, this is correct ✅", callback_data="confirm_breed"),
                        InlineKeyboardButton("No, this is incorrect ❌", callback_data="reject_breed"),
                        InlineKeyboardButton("I want to say what kind of breed it is 🐶", callback_data="want_to_say_breed"),
                        InlineKeyboardButton("I don't know what kind of breed it is 🪡", callback_data="dont_know"),
                        InlineKeyboardButton("Use the bot again 🐩", callback_data="start_over")
                    )
                    bot.send_message(
                        message.chat.id,
                        "Was this prediction correct?",
                        reply_markup=markup,
                    )
                elif 0.5<confidence<0.7:
                    try:
                        bot.delete_message(chat_id=message.chat.id, message_id=sent_message.message_id)
                    except telebot.apihelper.ApiTelegramException:
                        pass
                    bot.reply_to(
                        message,
                        f"It could be: {predicted_breed.replace('_', ' ')}, but I am not sure"
                    )

                    # Show feedback options
                    markup = InlineKeyboardMarkup()
                    markup.row_width = 1
                    markup.add(
                        InlineKeyboardButton("Yes, this is correct ✅", callback_data="confirm_breed"),
                        InlineKeyboardButton("No, this is incorrect ❌", callback_data="reject_breed"),
                        InlineKeyboardButton("I want to say what kind of breed it is 🐶", callback_data="want_to_say_breed"),
                        InlineKeyboardButton("I don't know what kind of breed it is 🪡", callback_data="dont_know"),
                        InlineKeyboardButton("Use the bot again 🐩", callback_data="start_over")
                    )
                    bot.send_message(
                        message.chat.id,
                        "Was this prediction correct?",
                        reply_markup=markup,
                    )
                else:
                    try:
                        bot.delete_message(chat_id=message.chat.id, message_id=sent_message.message_id)
                    except telebot.apihelper.ApiTelegramException:
                        pass
                    # bot.reply_to(
                    #     message,
                    #     f"I'm afraid to say the wrong breed because I'm not quite sure. Try to send me a better photo so I can tell for sure"
                    # )

                    # Show feedback options
                    markup = InlineKeyboardMarkup()
                    markup.row_width = 1
                    markup.add(
                        # InlineKeyboardButton("Yes, this is correct ✅", callback_data="confirm_breed"),
                        # InlineKeyboardButton("No, this is incorrect ❌", callback_data="reject_breed"),
                        # InlineKeyboardButton("I want to say what kind of breed it is 🐶", callback_data="want_to_say_breed"),
                        # InlineKeyboardButton("I don't know what kind of breed it is 🪡", callback_data="dont_know"),
                        InlineKeyboardButton("Use the bot again 🐩", callback_data="start_over")
                    )
                    bot.send_message(
                        message.chat.id,
                        f"I'm afraid to say the wrong breed because I'm not quite sure. Try to send me a better photo so I can tell for sure",
                        reply_markup=markup,
                    )
        except Exception as e:
            try:
                bot.delete_message(chat_id=message.chat.id, message_id=sent_message.message_id)
            except telebot.apihelper.ApiTelegramException:
                pass
            bot.reply_to(message, f"An error occurred during prediction: {e}")

@bot.callback_query_handler(func=lambda call: call.data in ["confirm_breed", "reject_breed", "want_to_say_breed", "dont_know"])
def handle_feedback(call):
    chat_id = call.message.chat.id
    try:
        bot.delete_message(chat_id, call.message.message_id)
    except telebot.apihelper.ApiTelegramException as e:
        if "message can't be deleted for everyone" in str(e):
            bot.answer_callback_query(call.id, "")

    if chat_id not in USER_DATA:
        bot.answer_callback_query(call.id, "No active prediction found.")
        return

    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Use the bot again 🐩", callback_data="start_over"))
            
    if call.data == "confirm_breed":
        bot.answer_callback_query(call.id, "Thank you for confirming!")
        bot.send_message(chat_id, "Great! I'm glad I got it right. 🐶", reply_markup=markup)
        image_path = USER_DATA[chat_id].get("image_path")
        predicted_breed = USER_DATA[chat_id]["predicted_breed"]
        if image_path and predicted_breed:
            save_directory = os.path.join(TRAINING_FOLDER, predicted_breed)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
            new_image_path = os.path.join(save_directory, os.path.basename(image_path))
            os.rename(image_path, new_image_path)
        del USER_DATA[chat_id]   

    elif call.data == "reject_breed":
        bot.answer_callback_query(call.id, "Thanks for your feedback!")
        bot.send_message(chat_id, "Please tell me the correct breed:")
        USER_DATA[chat_id]["awaiting_breed"] = True

    elif call.data == "want_to_say_breed":
        bot.answer_callback_query(call.id, "Got it!")
        USER_DATA[chat_id]["awaiting_breed"] = True
        bot.send_message(chat_id, "Thank you for joining our dataset ❤️ Please tell me the breed of your dog:")

    elif call.data == "dont_know":
        bot.answer_callback_query(call.id, "OK! I'll handle it.")
        bot.send_message(chat_id, "No worries! I'll take care of it. You can try again later. 💾", reply_markup=markup)
        del USER_DATA[chat_id]

@bot.message_handler(func=lambda message: USER_DATA.get(message.chat.id, {}).get("awaiting_breed", False))
def handle_correct_breed(message):
    chat_id = message.chat.id
    correct_breed = message.text.lower().replace(" ", "_")

    markup = InlineKeyboardMarkup()
    markup.row_width = 1
    markup.add(InlineKeyboardButton("Use the bot again 🐩", callback_data="start_over"))

    save_directory = os.path.join(TRAINING_FOLDER, correct_breed)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    image_path = USER_DATA[chat_id].get("image_path")
    if image_path:
        new_image_path = os.path.join(save_directory, os.path.basename(image_path))
        os.rename(image_path, new_image_path)

    bot.send_message(
        chat_id,
        f"Thanks for letting me know! I predicted: {USER_DATA[chat_id]['predicted_breed'].replace('_', ' ')}, "
        f"but the correct breed is: {correct_breed}."
    )
    bot.send_message(chat_id, "Now you can use this bot again to discover more dog breeds.", reply_markup=markup)
    del USER_DATA[chat_id]

if __name__ == '__main__':
    print("Bot started")
    
    while True:
        try:
            bot.polling(interval=1, timeout=40)
        except Exception as e:
            print(f"An error occurred: {e}")
            time.sleep(5)  # Wait before retrying

