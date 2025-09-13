import pygame
import random
import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import cohere
from dotenv import load_dotenv

load_dotenv()

pygame.init()

# Constants for the game
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
FONT = pygame.font.Font(None, 48)
HINT_DURATION = 10000

class WordPuzzleGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Word Puzzle Game with Dynamic SVM and Cohere")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Initialize Cohere client
        try:
            api_key = os.getenv('COHERE_API_KEY')
            if not api_key:
                raise ValueError("No Cohere API key found in .env file")
            self.co = cohere.Client(api_key)
            self.cohere_enabled = True
            print("Successfully initialized Cohere client")
        except Exception as e:
            print(f"Error initializing Cohere client: {e}")
            self.cohere_enabled = False
        
        print("Initializing Word Puzzle Game with SVM...")
        self.initialize_ml_model()
        self.reset_word_database()
        self.current_difficulty = None
        self.points = 0
        self.reset_game_state()
        self.word_pool = self.load_additional_words()
        
        self.current_screen = self.welcome_screen

    def reset_game_state(self):
        self.word = ""
        self.theme = "Crime Scene"
        self.guessed_letters = []
        self.attempts = 6
        self.max_revealed = 0
        self.hints_remaining = 3
        self.hint_message = ""
        self.hint_start_time = 0
        self.hint_cooldown = False
        self.hint_cooldown_start = 0

    def reset_word_database(self):
        self.word_database = {
            "Easy": ['murder', 'robbery', 'theft', 'clue', 'suspect'],
            "Medium": ['detective', 'evidence', 'witness', 'interrogate', 'alibi'],
            "Difficult": ['forensic', 'investigation', 'suspicious', 'perpetrator', 'casefile']
        }

    def load_additional_words(self):
        return [
            'crime', 'mystery', 'fingerprint', 'suspected', 'arrest', 'motive',
            'investigator', 'assassin', 'undercover', 'case', 'solution', 'whodunit'
        ]

    def initialize_ml_model(self):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
        self.svm_model = SVC(kernel='linear', probability=True)
        
        self.initial_training_data = {
            "Easy": ['murder', 'robbery', 'theft'],
            "Medium": ['detective', 'evidence', 'witness'],
            "Difficult": ['forensic', 'investigation', 'perpetrator']
        }
        
        if os.path.exists('model_data.pkl'):
            try:
                with open('model_data.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.training_data = data['training_data']
                    self.training_labels = data['training_labels']
                    self.vectorizer = data['vectorizer']
                    self.svm_model = data['svm_model']
                print(f"Loaded existing SVM model with {len(self.training_data)} training examples")
            except Exception as e:
                print(f"Error loading model data: {e}")
                self.initialize_default_training_data()
        else:
            self.initialize_default_training_data()
        
        difficulty_counts = {}
        for label in self.training_labels:
            difficulty_counts[label] = difficulty_counts.get(label, 0) + 1
        print("Current training data distribution:")
        for diff, count in difficulty_counts.items():
            print(f"  {diff}: {count} words")

    def initialize_default_training_data(self):
        self.training_data = []
        self.training_labels = []
        for difficulty, words in self.initial_training_data.items():
            for word in words:
                self.training_data.append(word)
                self.training_labels.append(difficulty)
        
        if len(set(self.training_labels)) > 1:
            X = self.vectorizer.fit_transform(self.training_data)
            self.svm_model.fit(X, self.training_labels)
        print("Initialized new SVM model with default training data")

    def save_model_data(self):
        data = {
            'training_data': self.training_data,
            'training_labels': self.training_labels,
            'vectorizer': self.vectorizer,
            'svm_model': self.svm_model
        }
        with open('model_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        print("Model data saved successfully")

    def update_model(self, word, difficulty):
        self.training_data.append(word)
        self.training_labels.append(difficulty)
        print(f"Adding word '{word}' to training data with difficulty '{difficulty}'")
        
        if len(set(self.training_labels)) > 1:
            X = self.vectorizer.fit_transform(self.training_data)
            self.svm_model.fit(X, self.training_labels)
            print(f"Training SVM with {len(self.training_data)} words")
        else:
            print("Not enough variety in training data, selecting difficulty randomly")
        
        self.save_model_data()

    def predict_difficulty(self, word):
        if len(set(self.training_labels)) <= 1:
            print("Not enough variety in training data, selecting difficulty randomly")
            return random.choice(list(self.word_database.keys()))
        
        X = self.vectorizer.transform([word])
        probabilities = self.svm_model.predict_proba(X)[0]
        predicted_difficulty = self.svm_model.predict(X)[0]
        
        print(f"Predicting difficulty for word '{word}'")
        for difficulty, prob in zip(self.svm_model.classes_, probabilities):
            print(f"  {difficulty}: {prob:.2f} probability")
        print(f"Final prediction: {predicted_difficulty}")
        
        return predicted_difficulty

    def display_message(self, message, x, y, justify="left", color=BLACK):
        text_surface = FONT.render(message, True, color)
        text_rect = text_surface.get_rect()
        if justify == "left":
            text_rect.topleft = (x, y)
        elif justify == "center":
            text_rect.center = (x, y)
        elif justify == "right":
            text_rect.topright = (x, y)
        self.screen.blit(text_surface, text_rect)

    def welcome_screen(self):
        self.screen.fill(WHITE)
        self.display_message("Welcome to the Crime Puzzle Game!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
        self.display_message("Press any key to start", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, justify="center")
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.current_screen = self.choose_difficulty

    def choose_difficulty(self):
        difficulties = list(self.word_database.keys())
        selected = 0
        choosing = True

        while choosing and self.running:
            self.screen.fill(WHITE)
            self.display_message("Select Difficulty:", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4, justify="center")
            for i, diff in enumerate(difficulties):
                color = BLACK if i == selected else GRAY
                self.display_message(f"{i + 1}. {diff}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + i * 60, justify="center", color=color)
            self.display_message("Use arrow keys to select, Enter to confirm", SCREEN_WIDTH // 2, SCREEN_HEIGHT - 100, justify="center", color=GRAY)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    choosing = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        selected = (selected - 1) % len(difficulties)
                    elif event.key == pygame.K_DOWN:
                        selected = (selected + 1) % len(difficulties)
                    elif event.key == pygame.K_RETURN:
                        self.current_difficulty = difficulties[selected]
                        self.start_game()
                        choosing = False

            self.clock.tick(30)

    def generate_word(self, target_difficulty):
        if target_difficulty not in self.word_database:
            return None

        available_words = self.word_database[target_difficulty] + self.word_pool
        if not available_words:
            return None

        word = random.choice(available_words)
        if word in self.word_database[target_difficulty]:
            self.word_database[target_difficulty].remove(word)
        predicted_difficulty = self.predict_difficulty(word)
        self.update_model(word, predicted_difficulty)
        return word

    def start_game(self):
        if self.current_difficulty is None or self.current_difficulty not in self.word_database:
            self.current_screen = self.choose_difficulty
            return

        self.reset_game_state()
        
        self.word = self.generate_word(self.current_difficulty)
        if not self.word:
            self.reset_word_database()
            self.word = self.generate_word(self.current_difficulty)
            if not self.word:
                self.display_message("Error: Unable to generate word. Please restart the game.", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, justify="center")
                pygame.display.flip()
                pygame.time.wait(2000)
                self.running = False
                return

        self.max_revealed = 3 if self.current_difficulty == "Easy" else (2 if self.current_difficulty == "Medium" else 1)
        revealed_letters = random.sample(list(self.word), min(self.max_revealed, len(self.word)))
        self.guessed_letters = list(revealed_letters)
        self.current_screen = self.game_screen

    def generate_hint_with_cohere(self):
        if not self.cohere_enabled:
            return f"A letter in the word is '{random.choice([l for l in self.word if l not in self.guessed_letters])}'"
    
        try:
            remaining_letters = [l for l in self.word if l not in self.guessed_letters]
            if not remaining_letters:
                return "You've already guessed all the letters!"
        
            prompt = f"""
        You are a clever game hint generator. The word is '{self.word}', which is related to crime scenes and detective work.
        The player has already guessed these letters: {', '.join(self.guessed_letters) if self.guessed_letters else 'none'}.
        
        Give a short, cryptic description of the word without using any unguessed letters.
        The description should be about what the word means or represents in a crime context.
        
        Rules for the hint:
        1. NEVER reveal any unguessed letters
        2. Focus on the word's meaning or role in crime scenes
        3. Keep it short (5-8 words)
        4. Make it clever and thought-provoking
        5. Don't use obvious synonyms
        
        Example good hints:
        - For 'detective': "Follows clues to catch bad guys"
        - For 'evidence': "Silent witness at the crime scene"
        - For 'alibi': "Your story of innocence and whereabouts"
        """
        
            response = self.co.generate(
                prompt=prompt,
                max_tokens=30,
                temperature=0.8,  # Slightly increased for more creative responses
                k=0,
                stop_sequences=[".", "\n"],
                return_likelihoods='NONE'
            )
        
            hint = response.generations[0].text.strip()
            return hint if hint else "Think about what happens at crime scenes"
        except Exception as e:
            print(f"Error generating hint with Cohere: {e}")
            return "Think about what happens at crime scenes"
    def use_hint(self):
        current_time = pygame.time.get_ticks()
        if self.hint_cooldown and current_time - self.hint_cooldown_start < 3000:
            return
        
        if self.hints_remaining > 0:
            self.hints_remaining -= 1
            self.hint_message = self.generate_hint_with_cohere()
            self.hint_start_time = current_time
            self.hint_cooldown = True
            self.hint_cooldown_start = current_time
        else:
            self.hint_message = "No hints remaining!"
            self.hint_start_time = current_time


    def game_screen(self):
        self.screen.fill(WHITE)
    
        self.display_message(f"Theme: {self.theme}", SCREEN_WIDTH // 8, SCREEN_HEIGHT // 8)
        self.display_message(f"Difficulty: {self.current_difficulty}", SCREEN_WIDTH // 8, SCREEN_HEIGHT // 8 + 50)
        self.display_message(f"Points: {self.points}", SCREEN_WIDTH - 200, SCREEN_HEIGHT - 50)
    
        display_word = ' '.join([letter if letter in self.guessed_letters else '_' for letter in self.word])
        self.display_message(display_word, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
    
        self.display_message(f"Attempts left: {self.attempts}", SCREEN_WIDTH // 8, SCREEN_HEIGHT // 1.5 + 50)
        self.display_message(f"Hints: {self.hints_remaining} (Press '0')", SCREEN_WIDTH // 8, SCREEN_HEIGHT // 1.5 + 100)
    
        current_time = pygame.time.get_ticks()
        if self.hint_message and current_time - self.hint_start_time < HINT_DURATION:  # Using the new constant
            words = self.hint_message.split()
            lines = []
            current_line = []
        
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 40:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
        
            if current_line:
                lines.append(' '.join(current_line))
        
            for i, line in enumerate(lines):
                self.display_message(line, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 1.5 + 200 + (i * 50), justify="center")
    
        if self.attempts == 0:
            if self.points >= 5:
                self.display_message("Press '1' to use 5 points for 3 more attempts", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 1.5 + 150, justify="center")
    
        pygame.display.flip()
    
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key >= pygame.K_a and event.key <= pygame.K_z:
                    self.guess_letter(chr(event.key))
                elif event.key == pygame.K_0:
                    self.use_hint()
                elif event.key == pygame.K_1 and self.attempts == 0 and self.points >= 5:
                    self.points -= 5
                    self.attempts = 3
    
        if set(self.word) <= set(self.guessed_letters):
            self.handle_win()
        elif self.attempts <= 0 and self.points < 5:
            self.handle_game_over()
    def handle_win(self):
        self.display_message(f"You guessed the word: {self.word}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, justify="center")
        pygame.display.flip()
        pygame.time.wait(2000)
        self.points += 10
        self.start_game()

    def handle_game_over(self):
        self.display_message(f"Game Over! The word was: {self.word}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, justify="center")
        self.display_message("Press any key to restart", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, justify="center")
        pygame.display.flip()
        self.wait_for_restart()

    def guess_letter(self, letter):
        if letter in self.guessed_letters:
            return
        if letter in self.word:
            self.guessed_letters.append(letter)
        else:
            if self.attempts > 0:
                self.attempts -= 1

    def wait_for_restart(self):
        waiting = True
        while waiting and self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    waiting = False
                elif event.type == pygame.KEYDOWN:
                    self.current_screen = self.choose_difficulty
                    self.reset_game_state()
                    waiting = False

    def run(self):
        while self.running:
            self.current_screen()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = WordPuzzleGame()
    game.run()
