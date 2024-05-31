import cv2
import mediapipe as mp
import pygame
import numpy as np
import sys
import os
import random

class Item:
    def __init__(self, x, y, image, type, game_screen):
        self.x = x
        self.y = y
        self.image = image
        self.item_type = type
        self.game_screen = game_screen
        self.grap = False
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def draw(self):
        self.game_screen.blit(self.image, self.rect.topleft)

    def update(self, new_x, new_y, hold):
        if hold == self.item_type:
            self.x = new_x
            self.y = new_y
            self.rect.topleft = (self.x, self.y)
            return self.item_type
        if hold == 0:
            x = new_x - self.rect.centerx
            y = new_y - self.rect.centery
            if np.sqrt(x**2 + y**2) < 80:
                self.grap = True
            else:
                self.grap = False
                return 0
            if self.grap:
                self.x = new_x
                self.y = new_y
                self.rect.topleft = (self.x, self.y)
                return self.item_type
        else:
            return hold

    def interaction(self, other):
        if isinstance(other, Trash):
            if self.item_type == other.trash_type:
                other.exist = False

class Trash:
    def __init__(self, x, y, image, type, game_screen):
        self.x = x
        self.y = y
        self.exist = True
        self.image = image
        self.trash_type = type
        self.game_screen = game_screen
        self.rect = self.image.get_rect(topleft=(self.x, self.y))

    def draw(self):
        if self.exist:
            self.game_screen.blit(self.image, self.rect.topleft)

def dist(x, y):
    return np.sqrt(x**2 + y**2)

def crash(A, B):
    x = A.rect.centerx - B.rect.centerx
    y = A.rect.centery - B.rect.centery
    return dist(x, y) < 40

cap = cv2.VideoCapture(0)
pygame.init()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
game_screen = pygame.display.set_mode((1280, 720))
pygame.display.set_caption("test")
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 배경 이미지 로드
current_path = os.path.dirname(os.path.abspath(__file__))
background_image_path = os.path.join(current_path, "image7.png")
background_image = pygame.image.load(background_image_path)
background_image = pygame.transform.scale(background_image, (1280, 720))

# 손 이미지 로드
hand_open_image_path = os.path.join(current_path, "handimage1.png")
hand_closed_image_path = os.path.join(current_path, "handimage2.png")
hand_image_open = pygame.image.load(hand_open_image_path)
hand_image_closed = pygame.image.load(hand_closed_image_path)
hand_image_open = pygame.transform.scale(hand_image_open, (70, 70))
hand_image_closed = pygame.transform.scale(hand_image_closed, (70, 70))

# 아이템 이미지 로드
broomstick_image_path = os.path.join(current_path, "broomstick.png")
can_image_path = os.path.join(current_path, "can.png")
paper_image_path = os.path.join(current_path, "paper.png")
broomstick_image = pygame.image.load(broomstick_image_path)
can_image = pygame.image.load(can_image_path)
paper_image = pygame.image.load(paper_image_path)
broomstick_image = pygame.transform.scale(broomstick_image, (100, 100))
can_image = pygame.transform.scale(can_image, (100, 100))
paper_image = pygame.transform.scale(paper_image, (100, 100))

# 쓰레기 이미지 로드
dust_image_path = os.path.join(current_path, "dust.png")
can2_image_path = os.path.join(current_path, "can2.png")
paper2_image_path = os.path.join(current_path, "paper2.png")
dust_image = pygame.image.load(dust_image_path)
can2_image = pygame.image.load(can2_image_path)
paper2_image = pygame.image.load(paper2_image_path)
dust_image = pygame.transform.scale(dust_image, (120, 120))
can2_image = pygame.transform.scale(can2_image, (120, 120))
paper2_image = pygame.transform.scale(paper2_image, (120, 120))

finish = False
clock = pygame.time.Clock()

# 랜덤한 좌표 생성
def random_position():
    x = random.randint(0, 1000)
    y = random.randint(0, 600)
    return x, y

# 아이템과 쓰레기의 초기화
item1 = Item(1100, 100, broomstick_image, 1, game_screen)
item2 = Item(1100, 300, can_image, 2, game_screen)
item3 = Item(1100, 500, paper_image, 3, game_screen)

trash1 = Trash(*random_position(), dust_image, 1, game_screen)
trash2 = Trash(*random_position(), dust_image, 1, game_screen)
trash3 = Trash(*random_position(), dust_image, 1, game_screen)
trash4 = Trash(*random_position(), can2_image, 2, game_screen)
trash5 = Trash(*random_position(), can2_image, 2, game_screen)
trash6 = Trash(*random_position(), paper2_image, 3, game_screen)
trash7 = Trash(*random_position(), paper2_image, 3, game_screen)

trashes = [trash1, trash2, trash3, trash4, trash5, trash6, trash7]
items = [item1, item2, item3]
hold = 0
prev_is_grabbing = False

def is_hand_closed(hand_landmarks, img_width, img_height):
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip_coords = (int(thumb_tip.x * img_width), int(thumb_tip.y * img_height))
    index_tip_coords = (int(index_tip.x * img_width), int(index_tip.y * img_height))
    distance = np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_tip_coords))
    return distance < 50

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_height, img_width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            finish = True

    game_screen.fill((0, 0, 0))
    game_screen.blit(background_image, (0, 0))

    for trash in trashes:
        trash.draw()
    for item in items:
        item.draw()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            hand_center = np.mean([[lmk.x * game_screen.get_width(), lmk.y * game_screen.get_height()] for lmk in hand_landmarks.landmark], axis=0)
            
            is_grabbing = is_hand_closed(hand_landmarks, img_width, img_height)
            if is_grabbing:
                hand_image = hand_image_closed
            else:
                hand_image = hand_image_open

            game_screen.blit(hand_image, (hand_center[0] - 30, hand_center[1] - 30))

            if hold != 0 and not is_grabbing and prev_is_grabbing:
                hold = 0  # Release the item when hand opens

            if hold == 0 and is_grabbing:
                for item in items:
                    hold = item.update(int(hand_center[0]), int(hand_center[1]), hold)
                    if hold != 0:
                        break
            elif hold == item1.item_type:
                hold = item1.update(int(hand_center[0]), int(hand_center[1]), hold)
            elif hold == item2.item_type:
                hold = item2.update(int(hand_center[0]), int(hand_center[1]), hold)
            elif hold == item3.item_type:
                hold = item3.update(int(hand_center[0]), int(hand_center[1]), hold)

            for item in items:
                for trash in trashes:
                    if crash(item, trash):
                        item.interaction(trash)

            prev_is_grabbing = is_grabbing

    pygame.display.flip()
    clock.tick(60)

    cv2.imshow('MediaPipe Hands', img)

    if cv2.waitKey(10) & 0xFF == 27 or finish:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
