import cv2
import mediapipe as mp
import pygame
import math
import numpy as np
import sys
import os

class Item:
    def __init__(self, x, y, color, type, game_screen):
        self.x = x
        self.y = y
        self.color = color
        self.item_type = type
        self.game_screen = game_screen
        self.grap = False
        self.rect = pygame.Rect(self.x, self.y, 60, 60)

    def draw(self):
        pygame.draw.rect(self.game_screen, self.color, self.rect)

    def update(self, new_x, new_y, dist_hand, hold):
        if hold == self.item_type:
            x = new_x - self.rect.centerx
            y = new_y - self.rect.centery
            if dist(x, y) < 80 and dist_hand < 80:
                self.grap = True
            else:
                self.grap = False
                return 0
            if self.grap:
                self.x = new_x
                self.y = new_y
                self.rect = pygame.Rect(self.x, self.y, 60, 60)
                return self.item_type
        if hold == 0:
            x = new_x - self.rect.centerx
            y = new_y - self.rect.centery
            if dist(x, y) < 80 and dist_hand < 80:
                self.grap = True
            else:
                self.grap = False
                return 0
            if self.grap:
                self.x = new_x
                self.y = new_y
                self.rect = pygame.Rect(self.x, self.y, 60, 60)
                return self.item_type
        else:
            return hold

    def interaction(self, other):
        if str(type(other)) == "<class '__main__.Trash'>":
            if self.item_type == other.trash_type:
                other.exist = False
        else:
            pass

class Trash:
    def __init__(self, x, y, color, type, game_screen):
        self.x = x
        self.y = y
        self.exist = True
        self.color = color
        self.trash_type = type
        self.game_screen = game_screen
        self.rect = pygame.Rect(self.x, self.y, 60, 60)

    def draw(self):
        if self.exist:
            pygame.draw.rect(self.game_screen, self.color, self.rect)
        else:
            pass

def dist(x, y):
    return math.sqrt(math.pow(x, 2) + math.pow(y, 2))

def crash(A, B):
    x = A.rect.centerx - B.rect.centerx
    y = A.rect.centery - B.rect.centery
    if dist(x, y) < 40:
        return True
    else:
        return False

cap = cv2.VideoCapture(0)
pygame.init()

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
game_screen = pygame.display.set_mode((1280, 720))  # 게임 스크린 크기 설정
pygame.display.set_caption("test")
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 배경 이미지 로드
current_path = os.path.dirname(os.path.abspath(__file__))
background_image_path = os.path.join(current_path, "image7.png")
background_image = pygame.image.load(background_image_path)
background_image = pygame.transform.scale(background_image, (1280, 720))

finish = False
clock = pygame.time.Clock()  # 주사율 설정용

item1 = Item(1000, 200, (0, 255, 255), 1, game_screen)
item2 = Item(1000, 400, (255, 0, 255), 2, game_screen)
item3 = Item(1000, 600, (255, 255, 0), 3, game_screen)

trash1 = Trash(200, 300, (0, 255, 255), 1, game_screen)
trash2 = Trash(400, 300, (255, 0, 255), 2, game_screen)
trash3 = Trash(600, 300, (255, 255, 0), 3, game_screen)

trashes = [trash1, trash2, trash3]
items = [item1, item2, item3]
hold = 0

while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)  # 좌우 반전
    img_height, img_width, _ = img.shape  # height 720 width 1280
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            finish = True

    game_screen.fill((0, 0, 0))  # 배경화면 업데이트
    game_screen.blit(background_image, (0, 0))  # 배경화면 그리기

    for trash in trashes:
        trash.draw()
    for item in items:
        item.draw()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # x 좌표는 좌우 반전 후 그대로 사용
            rect_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * game_screen.get_width())
            # y 좌표는 그대로 사용
            rect_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * game_screen.get_height())

            fin_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * game_screen.get_width())
            fin_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * game_screen.get_height())
            tmb_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * game_screen.get_width())
            tmb_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * game_screen.get_height())
            x = fin_tip_x - tmb_x
            y = fin_tip_y - tmb_y
            dist_hand = dist(x, y)
            if dist_hand <= 80:
                color = (255, 0, 0)
            else:
                color = (255, 255, 255)
            hand = pygame.Rect(rect_x, rect_y, 60, 60)
            pygame.draw.rect(game_screen, color, hand)  # 좌표에 따라 사각형 그리기

            if hold == 0:
                for item in items:
                    hold = item.update(fin_tip_x, fin_tip_y, dist_hand, hold)
                    if hold != 0:
                        break
            elif hold == item1.item_type:
                hold = item1.update(fin_tip_x, fin_tip_y, dist_hand, hold)
            elif hold == item2.item_type:
                hold = item2.update(fin_tip_x, fin_tip_y, dist_hand, hold)
            else:
                hold = item3.update(fin_tip_x, fin_tip_y, dist_hand, hold)

            for item in items:
                for trash in trashes:
                    if crash(item, trash):
                        item.interaction(trash)

    pygame.display.flip()
    clock.tick(60)  # 60fps

    # OpenCV 창에 손 이미지 표시 (디버깅 용도)
    cv2.imshow('MediaPipe Hands', img)

    if cv2.waitKey(10) & 0xFF == 27 or finish:
        break

cap.release()
cv2.destroyAllWindows()
pygame.quit()
