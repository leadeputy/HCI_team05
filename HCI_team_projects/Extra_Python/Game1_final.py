import cv2
import mediapipe as mp
import pygame
import numpy as np
import sys
import random
import os
import time

# 파이게임 불러오기 
pygame.init()

# 기본 화면 설정
screen_width, screen_height = 1280, 720
game_screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("HAN-DORM")

# 현재 스크립트의 디렉토리 경로를 가져오기
current_path = os.path.dirname(os.path.abspath(__file__))

# 인트로 화면 이미지
intro_images = [
    pygame.image.load(os.path.join(current_path, "image1.png")),
    pygame.image.load(os.path.join(current_path, "image2.png")),
    pygame.image.load(os.path.join(current_path, "image3.png")),
    pygame.image.load(os.path.join(current_path, "image4.png"))
]

# 게임1 끝난 이후 이미지 
next_images = [
    pygame.image.load(os.path.join(current_path, "NextGame.png")),
    pygame.image.load(os.path.join(current_path, "image5.png")),
    pygame.image.load(os.path.join(current_path, "image6.png"))
]

# Resize images to fit the game screen
intro_images = [pygame.transform.scale(image, (screen_width, screen_height)) for image in intro_images]
next_images = [pygame.transform.scale(image, (screen_width, screen_height)) for image in next_images]

# 버튼 만들기
buttons = [
    pygame.Rect(700, 390, 480, 200),  # "GAME START" on the first screen
    pygame.Rect(450, 435, 380, 200),  # "GAME START" on the second screen
    pygame.Rect(450, 425, 380, 200),  # "NEXT" on the third screen
    pygame.Rect(450, 425, 380, 200)   # "START" on the fourth screen
]

# 게임1 이후 화면의 버튼
next_buttons = [
    pygame.Rect(450, 425, 380, 200),  # "NEXT" on the NextGame.png screen
    pygame.Rect(450, 425, 380, 200),  # "NEXT" on the image5 screen
    pygame.Rect(450, 425, 380, 200)   # "START" on the image6 screen
]

# 게임1 배경화면 이미지
background_image_path = os.path.join(current_path, "image.png")
background_image = pygame.image.load(background_image_path)
background_image = pygame.transform.scale(background_image, (screen_width, screen_height))

# 클리어 이미지 로드 및 크기 조정
clear_image_path = os.path.join(current_path, "Clear2.png")
clear_image = pygame.image.load(clear_image_path)
clear_image = pygame.transform.scale(clear_image, (screen_width, screen_height))

# 손 이미지 로드
hand_image_open_path = os.path.join(current_path, "handimage1.png")
hand_image_closed_path = os.path.join(current_path, "handimage2.png")
hand_image_open = pygame.image.load(hand_image_open_path)
hand_image_open = pygame.transform.scale(hand_image_open, (60, 60))
hand_image_closed = pygame.image.load(hand_image_closed_path)
hand_image_closed = pygame.transform.scale(hand_image_closed, (60, 60))

# 미디어 파이프
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_tracking_confidence=0.5, min_detection_confidence=0.5)

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 쓰레기통 클래스 정의
class Trashbin:
    def __init__(self, x, y, image, type):
        self.x = x
        self.y = y
        self.image = image
        self.type = type

    def draw(self):
        game_screen.blit(self.image, (self.x, self.y))

# 아이템 클래스 정의
class Trash:
    def __init__(self, x, y, image, type):
        self.x = x
        self.y = y
        self.image = image
        self.type = type
        self.speed = 5  # 아이템이 내려오는 속도

    def move(self):
        self.y += self.speed
        if self.y > screen_height:
            self.y = 0
            self.x = random.randint(0, screen_width - self.image.get_width())

    def draw(self):
        game_screen.blit(self.image, (self.x, self.y))

    def follow_hand(self, hand_center):
        self.x = hand_center[0] - self.image.get_width() // 2
        self.y = hand_center[1] - self.image.get_height() // 2

# 쓰레기통 이미지 로드 및 생성
trashbin_list = ['general.png', 'plastic.png', 'can.png', 'paper.png']
trashbin_images = [pygame.transform.scale(pygame.image.load(os.path.join(current_path, f'game1/trashbin/{i}')), (150, 150)) for i in trashbin_list]

spacing = 300  # 쓰레기통 사이의 간격
initial_x = (screen_width - (len(trashbin_images) - 1) * spacing) // 2  # 쓰레기통이 화면 중앙에 위치하도록 초기 x 좌표 설정
trashbins = [Trashbin(initial_x + i * spacing, screen_height - img.get_height(), img, t.split('.')[0]) for i, (img, t) in enumerate(zip(trashbin_images, trashbin_list))]

# 쓰레기 아이템 이미지 로드 및 생성
trash_list = [
    ('bottle.png', 'plastic'), 
    ('carton.png', 'paper'), 
    ('milkcarton.png', 'paper'), 
    ('plbag.png', 'general'), 
    ('straw.png', 'plastic'), 
    ('can.png', 'can'), 
    ('can2.png', 'can'), 
    ('box.png', 'paper'), 
    ('plcup.png', 'plastic')
]

trash_images = [pygame.transform.scale(pygame.image.load(os.path.join(current_path, f'game1/item/{i[0]}')), (80, 80)) for i in trash_list]

# 각 쓰레기 이미지를 타입에 맞게 생성
random_trash = [Trash(random.randint(0, screen_width - img.get_width()), 0, img, t[1]) for img, t in zip(trash_images, trash_list)]

# 점수 변수
score = 0
holding_trash = None
current_screen = 0
last_gesture_time = time.time()
miss_time = 0

# 전역 변수 초기화
clock = pygame.time.Clock()
running = True

# 랜덤 아이템을 쓰레기통에 집어넣는 함수
def put_item_in_trashbin(item, trashbins):
    global score, holding_trash, miss_time
    for tb in trashbins:
        if tb.x < item.x < tb.x + tb.image.get_width() and tb.y < item.y < tb.y + tb.image.get_height():
            if tb.type == item.type:
                item.y = 0
                item.x = random.randint(0, screen_width - item.image.get_width())
                score += 1
                holding_trash = None
                return
            else:
                miss_time = time.time()
                holding_trash = None
                return

# 배경 이미지를 화면에 표시하는 함수
def display_background():
    game_screen.blit(background_image, (0, 0))

# 점수 표시 함수
def display_score(score):
    font = pygame.font.Font(None, 74)
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    game_screen.blit(score_text, (screen_width - 300, 10))

# 미스 표시 함수
def display_miss():
    font = pygame.font.Font(None, 74)
    miss_text = font.render("miss", True, (255, 0, 0))
    game_screen.blit(miss_text, (screen_width // 2 - 50, screen_height // 2 - 37))

# 클리어 화면 표시 함수
def display_clear_screen():
    game_screen.blit(clear_image, (0, 0))
    pygame.display.flip()
    pygame.time.wait(5000)

def handle_intro_screen():
    global current_screen, last_gesture_time, clock
    while current_screen < len(intro_images):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam")
            continue

        # 좌우 반전
        img = cv2.flip(img, 1)

        img_height, img_width, _ = img.shape  # height 720 width 1280
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw the current screen
        game_screen.blit(intro_images[current_screen], (0, 0))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 손의 특정 좌표를 가져옵니다.
                hand_center = np.mean([[lmk.x * screen_width, lmk.y * screen_height] for lmk in hand_landmarks.landmark], axis=0)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip_coords = (int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height))
                index_tip_coords = (int(index_tip.x * screen_width), int(index_tip.y * screen_height))
                
                distance_thumb_index = np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_tip_coords))
                
                if distance_thumb_index < 40:
                    hand_image = hand_image_closed
                else:
                    hand_image = hand_image_open

                game_screen.blit(hand_image, (hand_center[0] - 30, hand_center[1] - 30))

                # 현재 시간이 이전 제스처 인식 시간에서 x초 이상 지났는지 확인
                current_time = time.time()
                if current_time - last_gesture_time > 1.5:
                    # 버튼 영역을 감지
                    if buttons[current_screen].collidepoint(index_tip_coords):
                        # 손가락 끝과 엄지 손가락 끝의 거리 계산
                        if distance_thumb_index < 40:  # 손가락과 엄지 손가락이 가까이 있는 경우 (손을 쥐는 제스처)
                            print(f"Gesture detected: Moving to screen {current_screen + 1}")
                            current_screen += 1
                            last_gesture_time = current_time  # 제스처 인식 시간 업데이트
                            if current_screen >= len(intro_images):
                                return
            
                # 버튼 영역을 시각적으로 표시
                pygame.draw.rect(game_screen, (255, 0, 0), buttons[current_screen], 2)  # Red border for visibility

        # Update the display
        pygame.display.flip()

        # Display the webcam feed with OpenCV for debugging
        cv2.imshow('MediaPipe Hands', img)

        # Check for exit key
        if cv2.waitKey(10) & 0xFF == 27:
            pygame.quit()
            sys.exit()

        # Update the Pygame display
        clock.tick(60)  # 30fps로 제한

def handle_game_loop():
    global running, score, holding_trash, miss_time

    while running:
        # Pygame 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        success, img = cap.read()
        if not success:
            continue

        # 좌우 반전
        img = cv2.flip(img, 1)

        img_height, img_width, _ = img.shape  # height 720 width 1280
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 배경 이미지를 Pygame 창에 표시
        display_background()

        # trash 위치 업데이트 및 그리기
        for trash in random_trash:
            if trash != holding_trash:
                trash.move()
            trash.draw()

        # 쓰레기통 그리기
        for trashbin in trashbins:
            trashbin.draw()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 손의 특정 좌표를 가져옵니다.
                hand_center = np.mean([[lmk.x * screen_width, lmk.y * screen_height] for lmk in hand_landmarks.landmark], axis=0)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip_coords = (int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height))
                index_tip_coords = (int(index_tip.x * screen_width), int(index_tip.y * screen_height))
                
                distance_thumb_index = np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_tip_coords))
                
                if distance_thumb_index < 40:
                    hand_image = hand_image_closed
                else:
                    hand_image = hand_image_open

                game_screen.blit(hand_image, (hand_center[0] - 30, hand_center[1] - 30))

                # 손의 중심 좌표를 계산하여 아이템을 쓰레기통에 넣기
                for trash in random_trash:
                    if holding_trash is None and trash.x < hand_center[0] < trash.x + trash.image.get_width() and trash.y < hand_center[1] < trash.y + trash.image.get_height():
                        if distance_thumb_index < 40:  # 손가락과 엄지 손가락이 가까이 있는 경우 (손을 쥐는 제스처)
                            holding_trash = trash
                    if holding_trash:
                        holding_trash.follow_hand(hand_center)
                        put_item_in_trashbin(holding_trash, trashbins)
                        break

        # 점수 표시
        display_score(score)

        # miss 메시지 표시
        if miss_time > 0 and time.time() - miss_time < 1:
            display_miss()
        elif miss_time > 0 and time.time() - miss_time >= 1:
            miss_time = 0

        # Pygame 화면 업데이트
        pygame.display.flip()

        # OpenCV 창에 손 이미지 표시 (디버깅 용도)
        cv2.imshow('MediaPipe Hands', img)

        # 종료 키 확인
        if cv2.waitKey(10) & 0xFF == 27:
            running = False

        clock.tick(60)  # 60fps로 제한

        # 점수가 7점에 도달하면 클리어 화면 표시 후 종료
        if score >= 7:
            display_clear_screen()
            handle_next_images()
            return

def handle_next_images():
    global current_screen, last_gesture_time, clock

    current_screen = 0
    last_gesture_time = time.time()
    while current_screen < len(next_images):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        success, img = cap.read()
        if not success:
            print("Failed to capture image from webcam")
            continue

        # 좌우 반전
        img = cv2.flip(img, 1)

        img_height, img_width, _ = img.shape  # height 720 width 1280
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Draw the current screen
        game_screen.blit(next_images[current_screen], (0, 0))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 손의 특정 좌표를 가져옵니다.
                hand_center = np.mean([[lmk.x * screen_width, lmk.y * screen_height] for lmk in hand_landmarks.landmark], axis=0)
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip_coords = (int(thumb_tip.x * screen_width), int(thumb_tip.y * screen_height))
                index_tip_coords = (int(index_tip.x * screen_width), int(index_tip.y * screen_height))
                
                distance_thumb_index = np.linalg.norm(np.array(thumb_tip_coords) - np.array(index_tip_coords))
                
                if distance_thumb_index < 40:
                    hand_image = hand_image_closed
                else:
                    hand_image = hand_image_open

                game_screen.blit(hand_image, (hand_center[0] - 30, hand_center[1] - 30))

                # 현재 시간이 이전 제스처 인식 시간에서 1.5초 이상 지났는지 확인
                current_time = time.time()
                if current_time - last_gesture_time > 1.5:
                    # 버튼 영역을 감지
                    if next_buttons[current_screen].collidepoint(index_tip_coords):
                        if distance_thumb_index < 40:  # 손가락과 엄지 손가락이 가까이 있는 경우 (손을 쥐는 제스처)
                            print(f"Gesture detected: Moving to screen {current_screen + 1}")
                            current_screen += 1
                            last_gesture_time = current_time  # 제스처 인식 시간 업데이트
                            if current_screen >= len(next_images):
                                return
            
                # 버튼 영역을 시각적으로 표시
                pygame.draw.rect(game_screen, (255, 0, 0), next_buttons[current_screen], 2)  # Red border for visibility

        # Update the display
        pygame.display.flip()

        # Display the webcam feed with OpenCV for debugging
        cv2.imshow('MediaPipe Hands', img)

        # Check for exit key
        if cv2.waitKey(10) & 0xFF == 27:
            pygame.quit()
            sys.exit()

        # Update the Pygame display
        clock.tick(60)  # 30fps로 제한

# 실행 부분
handle_intro_screen()
running = True  # 게임 루프가 시작될 때 running 변수를 True로 설정
handle_game_loop()

# 자원 해제 및 종료
cap.release()
cv2.destroyAllWindows()
pygame.quit()
sys.exit()
