import pygame
import random
import sys
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer

# Initialize Pygame
pygame.init()

# Screen dimensions and colors
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = pygame.font.Font(None, 48)

class ClueScrambleGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Clue Scramble Game with SVM")
        self.clock = pygame.time.Clock()
        self.running = True
        self.current_screen = self.welcome_screen

        self.reset_word_database()  # Initialize word database first
        self.initialize_ml_model()   # Then initialize the ML model

        self.level = 0  # Current level
        self.total_levels = 3  # Total levels in the game
        self.descriptions = []  # To store descriptions of the culprit
        self.points = 10  # Start with 10 points
        self.current_clue_index = 0  # Track current clue index for sequential display

    def reset_word_database(self):
        # Limited set of words with descriptions
        self.word_database = {
            "clue": ("evidence", "A piece of information that helps to solve a crime."),
            "motive": ("reason", "The reason someone would commit a crime."),
            "suspect": ("person", "A person who might have committed the crime."),
            "weapon": ("gun", "An object used to inflict harm."),
            "alibi": ("excuse", "A claim that one was elsewhere when the crime occurred."),
            "thief": ("criminal", "A person who steals."),
            "robbery": ("theft", "The action of taking property unlawfully."),
            "evidence": ("proof", "Anything that helps to prove something."),
            "investigate": ("explore", "To look into something carefully."),
            "detective": ("investigator", "A person who investigates crimes."),
        }

    def initialize_ml_model(self):
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2))
        self.svm_model = SVC(kernel='linear', probability=True)
        self.training_data = []
        self.training_labels = []

        # Prepare training data from the limited word database
        for word, (hint, _) in self.word_database.items():
            self.training_data.append(word)
            self.training_labels.append(hint)

        X = self.vectorizer.fit_transform(self.training_data)
        self.svm_model.fit(X, self.training_labels)

    def predict_clue(self, word):
        X = self.vectorizer.transform([word])
        similar_word = self.svm_model.predict(X)[0]
        return similar_word

    def scramble_word(self, word):
        word = list(word)
        random.shuffle(word)
        return ''.join(word)

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
                self.current_screen = self.choose_clue

    def choose_clue(self):
        if self.level < self.total_levels:
            # Randomly choose three clues for the current level
            self.current_clues = random.sample(list(self.word_database.keys()), 3)
            self.hint_sentences = [
                self.word_database[self.current_clues[0]][1],
                self.word_database[self.current_clues[1]][1],
                self.word_database[self.current_clues[2]][1]
            ]
            self.scrambled_words = [self.scramble_word(clue) for clue in self.current_clues]
            self.attempts = 5  # Reset attempts for the new round
            self.guessed_letters = []
            self.current_clue_index = 0  # Reset clue index for new level
            self.current_screen = self.game_screen
        else:
            self.current_screen = self.show_story

    def game_screen(self):
        self.screen.fill(WHITE)
        self.display_message(f"Level: {self.level + 1}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4, justify="center")

        # Show the current scrambled word
        if self.current_clue_index < len(self.scrambled_words):
            scrambled_word = self.scrambled_words[self.current_clue_index]
            self.display_message(f"Scrambled Word: {scrambled_word}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4 + 50, justify="center")

        self.display_message(f"Attempts left: {self.attempts}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 100, justify="center")
        self.display_message(f"Points: {self.points}", SCREEN_WIDTH - 200, SCREEN_HEIGHT - 50, justify="right")

        # Display the guessed letters
        guessed_word = ''.join(self.guessed_letters)
        self.display_message(f"Guessed Letters: {guessed_word}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
        self.display_message("Press 0 for a hint", 50, SCREEN_HEIGHT - 50, justify="left")  # Moved to bottom left corner

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:  # Check for word submission
                    self.submit_guess()
                elif event.key == pygame.K_BACKSPACE:  # Backspace functionality
                    if self.guessed_letters:  # Remove last letter if any
                        self.guessed_letters.pop()
                elif event.key >= pygame.K_a and event.key <= pygame.K_z:
                    letter = chr(event.key)
                    self.guess_letter(letter)
                elif event.key == pygame.K_0:  # Hint functionality via 0 key
                    self.show_hint()

        if self.attempts <= 0:
            self.display_message(f"Game Over! The clues were: {', '.join(self.current_clues)}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100, justify="center")
            pygame.display.flip()
            pygame.time.wait(2000)
            self.current_screen = self.choose_clue

    def guess_letter(self, letter):
        self.guessed_letters.append(letter) 

    def submit_guess(self):
        full_word = ''.join(self.guessed_letters)
        if full_word in self.current_clues:
            self.points += 10  # Award points for guessing the correct clue
            self.descriptions.append(self.hint_sentences[self.current_clues.index(full_word)])  # Add description for the current clue
            self.current_clue_index += 1  # Move to the next clue

        # Clear the guessed letters for the next word
            self.guessed_letters = []

            if self.current_clue_index < len(self.scrambled_words):
                self.current_screen = self.show_next_word  # Show next word screen
            else:
                self.level += 1  # Move to the next level after all clues
                self.current_clue_index = 0  # Reset clue index for next level
                self.current_screen = self.choose_clue
        else:
            self.attempts -= 1


    def show_next_word(self):
        self.screen.fill(WHITE)
        self.display_message(f"Congratulations! You found the clue: {self.current_clues[self.current_clue_index - 1]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
        self.display_message("Press any key for the next word.", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, justify="center")
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.current_screen = self.game_screen  # Go back to game screen for the next word

    def show_hint(self):
        if self.points >= 5:  # Ensure player has enough points for a hint
            self.points -= 5
            self.display_message(f"Hint: {self.hint_sentences[self.current_clue_index]}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, justify="center")
            pygame.display.flip()
            pygame.time.wait(2000)  # Wait for a few seconds to let the player see the hint

    def show_story(self):
        self.screen.fill(WHITE)
        self.display_message("Final Description of the Culprit:", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4, justify="center")
        
        # Display all descriptions collected during the game
        for index, description in enumerate(self.descriptions):
            self.display_message(f"Description {index + 1}: {description}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4 + 50 * index, justify="center")

        # Generate and display a new random culprit name
        culprit_name = self.generate_culprit_name()
        self.display_message(f"The culprit is: {culprit_name}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4 + 50 * (len(self.descriptions) + 1), justify="center")

        self.display_message("Press any key to play again", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100, justify="center")
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                self.level = 0  # Reset level
                self.descriptions = []  # Clear descriptions
                self.guessed_letters = []  # Clear guessed letters
                self.current_screen = self.welcome_screen  # Go back to the welcome screen

    def generate_culprit_name(self):
        names = ["John Doe", "Jane Smith", "Alex Brown", "Emily White", "Michael Black"]
        return random.choice(names)

    def run(self):
        while self.running:
            self.current_screen()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    game = ClueScrambleGame()
    game.run()
