import pygame
import random
import sys
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from cohere import Client
import os
from dotenv import load_dotenv

pygame.init()

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = pygame.font.Font(None, 48)

class ClueScrambleGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Clue Scramble Game with SVM and Cohere")
        self.clock = pygame.time.Clock()
        self.running = True
        self.current_screen = self.welcome_screen

        self.initialize_cohere()
        self.initialize_ml_model()

        self.current_level = 1
        self.max_levels = 3
        self.culprit_hints = []
        self.current_word_index = 0
        self.guessed_letters = []
        self.attempts = 5
        self.points = 10
        
        # List to keep track of used words
        self.used_words = set()

    def initialize_cohere(self):
        load_dotenv()
        api_key = os.getenv('COHERE_API_KEY')
        if api_key:
            self.cohere_client = Client(api_key)
        else:
            print("Cohere API key not found in .env file. Please add COHERE_API_KEY to your .env file.")
            sys.exit(1)

    def initialize_ml_model(self):
        # Expanded word list for better variety
        self.word_categories = {
            'mystery_words': [
                "alibi", "clue", "crime", "deduce", "detective", "evidence",
                "fingerprint", "investigate", "motive", "murder", "mystery",
                "scene", "sleuth", "solve", "suspect", "testimony", "theft",
                "trace", "victim", "witness"
            ],
            'location_words': [
                "attic", "basement", "bedroom", "cellar", "den", "garage",
                "garden", "hallway", "kitchen", "library", "lounge", "mansion",
                "office", "parlor", "patio", "room", "study", "theater"
            ],
            'weapon_words': [
                "blade", "candlestick", "dagger", "gun", "knife", "lead",
                "pipe", "poison", "revolver", "rope", "syringe", "toxin",
                "trap", "wrench"
            ]
        }
        
        # Combine all words for training
        all_words = [word for category in self.word_categories.values() for word in category]
        
        self.word_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
        self.word_svm = SVC(kernel='linear', probability=True)
        
        X_words = self.word_vectorizer.fit_transform(all_words)
        self.word_svm.fit(X_words, all_words)

    def generate_word(self):
        # Choose a category based on the current word index
        categories = list(self.word_categories.keys())
        current_category = categories[self.current_word_index % len(categories)]
        
        available_words = [word for word in self.word_categories[current_category] 
                          if word not in self.used_words]
        
        if not available_words:
            # If all words in the category are used, reset for that category
            available_words = self.word_categories[current_category]
            self.used_words -= set(available_words)
        
        # Choose a random word from the available words
        word = random.choice(available_words)
        self.used_words.add(word)
        
        return word

    def scramble_word(self, word):
        word = list(word)
        random.shuffle(word)
        return ''.join(word)

    def generate_culprit_description(self):
        try:
            prompts = [
    "Describe the suspect's possible motive in a single, suspenseful sentence.",
    "Write a brief and intriguing description of the weapon the culprit might have used.",
    "Compose a mysterious sentence hinting at the time or setting of the crime.",
    "Provide a short, captivating clue about a potential witness or accomplice involved in the case."
                ]
            prompt = prompts[len(self.culprit_hints) % len(prompts)]
            
            response = self.cohere_client.generate(
                model='command-nightly',
                prompt=prompt,
                max_tokens=50,
                temperature=0.8,
                k=0,
                stop_sequences=["."],
                return_likelihoods='NONE')
            
            # Ensure the response ends with a period
            generated_text = response.generations[0].text.strip()
            if not generated_text.endswith('.'):
                generated_text += '.'
            
            return generated_text
        except Exception as e:
            print(f"Error generating Cohere description: {e}")
            return "The suspect remains mysterious and leaves more questions than answers."

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
        self.display_message("Welcome to Clue Scramble Game!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
        self.display_message("Press any key to start", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, justify="center")
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.current_screen = self.level_intro_screen

    def level_intro_screen(self):
        self.screen.fill(WHITE)
        self.display_message(f"Level {self.current_level}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
        self.display_message("Press any key to continue", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, justify="center")
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.current_word_index = 0
                self.choose_clue()

    def choose_clue(self):
        if self.current_word_index < 3:
            self.current_clue = self.generate_word()
            self.scrambled_word = self.scramble_word(self.current_clue)
            self.attempts = 5
            self.guessed_letters = []
            self.current_screen = self.game_screen
        else:
            self.culprit_hints.append(self.generate_culprit_description())
            self.current_screen = self.culprit_hint_screen

    def game_screen(self):
        self.screen.fill(WHITE)
        self.display_message(f"Level {self.current_level} - Word {self.current_word_index + 1}", SCREEN_WIDTH // 2, 50, justify="center")
        self.display_message("Scrambled Word: " + self.scrambled_word, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4, justify="center")
        self.display_message(f"Attempts left: {self.attempts}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100, justify="center")
        self.display_message(f"Points: {self.points}", SCREEN_WIDTH - 200, SCREEN_HEIGHT - 50, justify="right")

        guessed_word = ''.join(self.guessed_letters)
        self.display_message(f"Guessed Letters: {guessed_word}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
        self.display_message("Press 0 for a hint", 50, SCREEN_HEIGHT - 50, justify="left")

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    self.submit_guess()
                elif event.key == pygame.K_BACKSPACE:
                    if self.guessed_letters:
                        self.guessed_letters.pop()
                elif event.key >= pygame.K_a and event.key <= pygame.K_z:
                    letter = chr(event.key)
                    self.guess_letter(letter)
                elif event.key == pygame.K_0:
                    self.show_hint()

    def guess_letter(self, letter):
        if len(self.guessed_letters) < len(self.current_clue):
            self.guessed_letters.append(letter)

    def submit_guess(self):
        guessed_word = ''.join(self.guessed_letters)
        if guessed_word == self.current_clue:
            self.points += 10
            self.display_message(f"Correct! The word was: {self.current_clue}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100, justify="center")
            pygame.display.flip()
            pygame.time.wait(2000)
            self.current_word_index += 1
            self.guessed_letters = []
            self.choose_clue()
        else:
            self.attempts -= 1
            self.guessed_letters = []
            if self.attempts <= 0:
                self.display_message(f"Out of attempts! The word was: {self.current_clue}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100, justify="center")
                pygame.display.flip()
                pygame.time.wait(2000)
                self.current_word_index += 1
                self.choose_clue()

    def show_hint(self):
        if self.points >= 5:
            self.points -= 5
            hint = self.generate_cohere_hint(self.current_clue)
            
            self.screen.fill(WHITE)
            words = hint.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 40:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            self.display_message("Hint:", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 75, justify="center")
            for i, line in enumerate(lines):
                self.display_message(line, SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 25 + (i * 50), justify="center")
            
            pygame.display.flip()
            pygame.time.wait(6000)
        else:
            self.display_message("Not enough points for a hint!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, justify="center")
            pygame.display.flip()
            pygame.time.wait(2000)

    def generate_cohere_hint(self, word):
        try:
            response = self.cohere_client.generate(
                model='command-nightly',
                prompt=f"Give a clever one-sentence hint for the word '{word}' without using the word itself:",
                max_tokens=30,
                temperature=0.7,
                k=0,
                stop_sequences=[],
                return_likelihoods='NONE')
            return response.generations[0].text.strip()
        except Exception as e:
            print(f"Error generating Cohere hint: {e}")
            return f"This {len(word)}-letter word might be important to solve the mystery..."

    def culprit_hint_screen(self):
        self.screen.fill(WHITE)
        
        title_y = 30
        hint_y = 90 
        
        self.display_message(f"Level {self.current_level} Complete!", SCREEN_WIDTH // 2, title_y, justify="center")
        
        # Get the current hint and word wrap it
        hint = self.culprit_hints[-1]
        words = hint.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 60:
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        hint_font = pygame.font.Font(None, 40)
        
        for line in lines:
            text_surface = hint_font.render(line, True, BLACK)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, hint_y))
            self.screen.blit(text_surface, text_rect)
            hint_y += 35  # Adjust line spacing as needed
        

        self.display_message("Press any key to continue", SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50, justify="center")
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.current_level += 1
                if self.current_level <= self.max_levels:
                    self.current_screen = self.level_intro_screen
                else:
                    self.current_screen = self.game_over_screen
    def game_over_screen(self):
        self.screen.fill(WHITE)
        
        title_y = 50
        score_y = 100
        story_y = 150
        
        self.display_message("Congratulations! You've completed all levels!", SCREEN_WIDTH // 2, title_y, justify="center")
        self.display_message(f"Final Score: {self.points}", SCREEN_WIDTH // 2, score_y, justify="center")
        self.display_message("The Complete Mystery:", SCREEN_WIDTH // 2, story_y, justify="center")
        
        # Combine all hints into one coherent paragraph
        full_story = " ".join(self.culprit_hints)
        
        # Word wrap the full story
        words = full_story.split()
        lines = []
        current_line = []
        
        for word in words:
            current_line.append(word)
            if len(' '.join(current_line)) > 60:   
                lines.append(' '.join(current_line[:-1]))
                current_line = [word]
        if current_line:
            lines.append(' '.join(current_line))
        
        
        story_font = pygame.font.Font(None, 36)  
        y_offset = story_y + 50
        
        for line in lines:
            text_surface = story_font.render(line, True, BLACK)
            text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, y_offset))
            self.screen.blit(text_surface, text_rect)
            y_offset += 30
            
        exit_font = pygame.font.Font(None, 36)
        exit_y = max(y_offset + 40, SCREEN_HEIGHT - 30)
        
        exit_text = exit_font.render("Press any key to exit", True, BLACK)
        exit_rect = exit_text.get_rect(center=(SCREEN_WIDTH // 2, exit_y))
        self.screen.blit(exit_text, exit_rect)
        
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN:
                self.running = False

    def run(self):
        while self.running:
            self.current_screen()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = ClueScrambleGame()
    game.run()
