import pygame
import random
import sys
import heapq

#Initialize Pygame
pygame.init()

#Constants specifications
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = pygame.font.Font(None, 48)

#Levels and words
LEVELS = {
    "Easy": {
        "levels": ["Level 1", "Level 2", "Level 3"],
        "levels_data": {
            "Level 1": {"words": ['python', 'java', 'ruby', 'html', 'css', 'php', 'csharp', 'perl', 'swift', 'sql'], 
                        "theme": "Popular Programming Languages", "max_revealed": 4},
            "Level 2": {"words": ['javascript', 'typescript', 'scala', 'rust', 'kotlin', 'assembly', 'bash', 'go', 'lua'], 
                        "theme": "Languages Used in Web Development", "max_revealed": 3},
            "Level 3": {"words": ['algorithm', 'network', 'database', 'api', 'json', 'ajax', 'microservices', 'docker', 'kubernetes'], 
                        "theme": "Key Concepts in Computer Science", "max_revealed": 2}
        }
    },
    "Medium": {
        "levels": ["Level 1", "Level 2", "Level 3"],
        "levels_data": {
            "Level 1": {"words": ['react', 'angular', 'nodejs', 'express', 'mongodb', 'vue', 'ember', 'backbone', 'jquery'], 
                        "theme": "Frontend Frameworks and Libraries", "max_revealed": 4},
            "Level 2": {"words": ['node', 'rest', 'soap', 'graphql', 'websocket', 'mqtt', 'tcp', 'udp', 'http'], 
                        "theme": "Protocols and Communication in Backend", "max_revealed": 3},
            "Level 3": {"words": ['graphql', 'apollo', 'nextjs', 'nuxt', 'webpack', 'babel', 'jest', 'mocha', 'chai'], 
                        "theme": "Tools and Testing in Web Development", "max_revealed": 2}
        }
    },
    "Difficult": {
        "levels": ["Level 1", "Level 2", "Level 3"],
        "levels_data": {
            "Level 1": {"words": ['machinelearning', 'neuralnetwork', 'deeplearning', 'reinforcementlearning', 'supervisedlearning', 'unsupervisedlearning', 'semi-supervisedlearning', 'activelearning', 'ensemblelearning', 'decisiontree'], 
                        "theme": "Advanced Concepts in Artificial Intelligence", "max_revealed": 4},
            "Level 2": {"words": ['cybersecurity', 'encryption', 'firewall', 'authentication', 'authorization', 'networksecurity', 'websecurity', 'endpointsecurity', 'datasecurity'], 
                        "theme": "Security Measures and Protocols", "max_revealed": 3},
            "Level 3": {"words": ['microservices', 'devops', 'continuousintegration', 'continuousdelivery', 'continuousdeployment', 'infrastructureascode', 'configurationmanagement', 'containerization', 'orchestration'], 
                        "theme": "Practices and Tools in DevOps", "max_revealed": 2}
        }
    }
}

#Helper function for A* algorithm
def heuristic(start, goal):
    return abs(ord(start) - ord(goal))

#A* algorithm implementation
def a_star(graph, start, goal, heuristic):
    # Initialize open and closed lists
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {}

    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0

    f_score = {node: float('inf') for node in graph}
    f_score[start] = heuristic(start, goal)

    # A* algorithm loop
    while open_list:
        current_node = heapq.heappop(open_list)[1]
        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            return path

        for next_node in graph[current_node]:
            tentative_g_score = g_score[current_node] + heuristic(current_node, next_node)
            if tentative_g_score < g_score[next_node]:
                came_from[next_node] = current_node
                g_score[next_node] = tentative_g_score
                f_score[next_node] = g_score[next_node] + heuristic(next_node, goal)
                heapq.heappush(open_list, (f_score[next_node], next_node))

    return None

class WordPuzzleGame:
    
    def __init__(self):
        self.initialize_pygame()
        self.current_difficulty = None
        self.current_level = None
        self.word = ""
        self.theme = ""
        self.guessed_letters = []
        self.attempts = 6
        self.max_revealed = 0
        self.hints_remaining = 5
        self.hint_message = ""
        self.hint_start_time = 0
        self.undo_stack = []
        self.game_over = False
        self.points = 0
        self.welcome_screen()

    def initialize_pygame(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Word Puzzle Game")
        self.clock = pygame.time.Clock()

    def welcome_screen(self):
        self.screen.fill(WHITE)
        self.display_message("Welcome to Byte Brain!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
        self.display_message("Let's get started..", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, justify="center")
        pygame.display.flip()
        pygame.time.delay(5000)  # Display the welcome message for 5 seconds
        self.choose_difficulty()

    def choose_difficulty(self):
        selected_difficulty = None
        while selected_difficulty is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        selected_difficulty = "Easy"
                    elif event.key == pygame.K_2:
                        selected_difficulty = "Medium"
                    elif event.key == pygame.K_3:
                        selected_difficulty = "Difficult"

            if selected_difficulty:
                self.current_difficulty = selected_difficulty
                self.choose_level()

            self.screen.fill(WHITE)
            self.display_message("Select Difficulty:", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4, justify="center")
            self.display_message("1. Easy", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, justify="center")
            self.display_message("2. Medium", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 60, justify="center")
            self.display_message("3. Difficult", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 120, justify="center")
            pygame.display.flip()

    def choose_level(self):
        selected_level = None
        levels = LEVELS[self.current_difficulty]["levels"]
        while selected_level is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key >= pygame.K_1 and event.key <= pygame.K_3:
                        level_index = event.key - pygame.K_1
                        if level_index < len(levels):
                            selected_level = levels[level_index]

            if selected_level:
                self.current_level = selected_level
                self.start_game()

            self.screen.fill(WHITE)
            self.display_message(f"Select Level - {self.current_difficulty}:", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4, justify="center")
            for i, level in enumerate(levels):
                self.display_message(f"{i + 1}. {level}", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + i * 60, justify="center")
            pygame.display.flip()

    def start_game(self):
        level_data = LEVELS[self.current_difficulty]["levels_data"][self.current_level]
        self.current_word_data = {
            "word": random.choice(level_data["words"]),
            "theme": level_data["theme"],
            "max_revealed": level_data["max_revealed"]
        }
        self.word = self.current_word_data["word"]
        self.theme = self.current_word_data["theme"]
        self.max_revealed = self.current_word_data["max_revealed"]
        revealed_letters = random.sample(self.word, random.randint(1, self.max_revealed))
        self.guessed_letters = list(revealed_letters)
        self.undo_stack = []
        self.game_over = False
        self.attempts = 6  # Reset attempts count to its initial value
        self.main_loop()

    def guess_letter(self, letter):
        if letter not in self.guessed_letters:
            self.guessed_letters.append(letter)
            if letter not in self.word:
                self.attempts -= 1

    def use_hint(self):
        if self.hints_remaining > 0:
            unguessed_letters = [letter for letter in set(self.word) if letter not in self.guessed_letters]
            if unguessed_letters:
                revealed_letter = random.choice(unguessed_letters)
                self.guessed_letters.append(revealed_letter)
                self.hints_remaining -= 1
                self.hint_message = f"The revealed letter is '{revealed_letter}'"
                self.hint_start_time = pygame.time.get_ticks()
            else:
                self.hint_message = "No more unguessed letters left."
                self.hint_start_time = pygame.time.get_ticks()
        else:
            self.hint_message = "No more hints remaining."
            self.hint_start_time = pygame.time.get_ticks()

    def display_word(self, y):
        display_word = ""
        for letter in self.word:
            if letter in self.guessed_letters:
                display_word += letter
            else:
                display_word += "_"

        
        text_y = min(y, SCREEN_HEIGHT - 50)
        self.display_message(display_word, SCREEN_WIDTH // 2, text_y, justify="center")

    def display_message(self, message, x, y, justify="left"):
        text_surface = FONT.render(message, True, BLACK)
        text_rect = text_surface.get_rect()
        if justify == "left":
            text_rect.topleft = (x, y)
        elif justify == "center":
            text_rect.center = (x, y)
        elif justify == "right":
            text_rect.topright = (x, y)
        self.screen.blit(text_surface, text_rect)

    def main_loop(self):
        while not self.game_over:
            word_guessed = ''.join(char for char in self.word if char in self.guessed_letters) == self.word

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key >= pygame.K_a and event.key <= pygame.K_z:
                        self.guess_letter(chr(event.key))
                    elif event.key == pygame.K_0:
                        self.use_hint()

            self.screen.fill(WHITE)
    
            theme_x = max(0, min(SCREEN_WIDTH // 8, SCREEN_WIDTH - 200))
            theme_y = max(0, min(SCREEN_HEIGHT // 8, SCREEN_HEIGHT - 50))
            self.display_message(f"Theme: {self.theme}", theme_x, theme_y, justify="left")
            self.display_message(f"Points: {self.points}", SCREEN_WIDTH - 200, SCREEN_HEIGHT - 50, justify="right")
            self.display_word(SCREEN_HEIGHT // 2)
            self.display_message(f"Attempts left: {self.attempts}", SCREEN_WIDTH // 8, SCREEN_HEIGHT // 1.5 + 50, justify="left")
            self.display_message(f"Hints: {self.hints_remaining} (Key'0')", SCREEN_WIDTH // 8, SCREEN_HEIGHT // 1.5 + 100, justify="left")
            if self.hint_message:
                self.display_message(self.hint_message, SCREEN_WIDTH // 8, SCREEN_HEIGHT // 1.5 + 150, justify="left")
                if pygame.time.get_ticks() - self.hint_start_time >= 2000:
                    self.hint_message = ""
            pygame.display.flip()

            if word_guessed:
                self.display_congratulatory_message()
            elif self.attempts == 0:
                self.display_game_over_message()

            self.clock.tick(60)

    def display_congratulatory_message(self):
        self.game_over = True
        self.points += 10
        self.screen.fill(WHITE)
        self.display_message("Congratulations! You guessed the word!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, justify="center")
        pygame.display.flip()
        pygame.time.delay(3000)
        self.choose_level()

    def display_game_over_message(self):
        self.game_over = True
        self.screen.fill(WHITE)
        self.display_message("Game Over! You've lost all your attempts.", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 50, justify="center")
        self.display_message("Press '2' to use points for more attempts.", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50, justify="center")
        self.display_message("Press any other key to exit.", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 100, justify="center")
        pygame.display.flip()
        pygame.event.clear()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_2:
                        if self.points >= 5:
                            self.points -= 5
                            self.attempts += 3
                            # Start the game again with the stored word and its associated data
                            self.word = self.current_word_data["word"]
                            self.theme = self.current_word_data["theme"]
                            self.max_revealed = self.current_word_data["max_revealed"]
                            self.guessed_letters = []
                            revealed_letters = random.sample(self.word, random.randint(1, self.max_revealed))
                            self.guessed_letters = list(revealed_letters)
                            self.undo_stack = []
                            self.game_over = False
                            self.main_loop()
                        else:
                            self.display_message("Insufficient points!", SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 150, justify="center")
                            pygame.display.flip()
                            pygame.time.delay(2000)
                            self.choose_level()
                    else:
                        self.choose_level()


if __name__ == "__main__":
    game = WordPuzzleGame()
