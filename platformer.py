"""
Simple pixel-perfect platformer.

Usage:
    python platformer.py [background.png] [mask.png]

- background.png : the scene (the window is sized to match it)
- mask.png       : binary mask. White (or any bright, opaque) pixels are solid
                   platform; black / transparent pixels are empty air.
                   It is scaled to the background's size if they differ.

If the files aren't found, a built-in demo scene is used so it runs out of the box.

Controls:  Left/Right or A/D = move,  Space/Up/W = jump,  M = toggle mask overlay,  Esc = quit
"""
import sys
import pygame
import numpy as np

# ---------------------------------------------------------------- settings ---
BG_PATH = sys.argv[1] if len(sys.argv) > 1 else "background.png"
MASK_PATH = sys.argv[2] if len(sys.argv) > 2 else "mask.png"

FPS = 60
SHOW_MASK = True                       # overlay the mask on the background (toggle with M)
MASK_TINT = (255, 70, 70, 110)         # RGBA colour of the overlay

SPRITE_PATH = sys.argv[3] if len(sys.argv) > 3 else "cat2.png"
SPRITE_HEIGHT = 68                     # on-screen height in px (width keeps aspect ratio)
SPRITE_FACES_RIGHT = True              # set False if your art faces left
FALLBACK_W, FALLBACK_H = 20, 32        # used if no sprite is found
MOVE_SPEED = 4.0
GRAVITY = 0.55
JUMP_SPEED = 14.5
MAX_FALL = 14.0
MAX_STEP = 4                           # tallest ledge / slope (in px) you can walk up


# ------------------------------------------------------------- scene setup ---
def make_demo_scene(size=(960, 540)):
    """Fallback scene: returns (background_surface, mask_surface)."""
    w, h = size
    bg = pygame.Surface(size)
    for y in range(h):                                   # sky gradient
        t = y / h
        pygame.draw.line(bg, (int(90 + 120 * t), int(150 + 80 * t), 235), (0, y), (w, y))
    pygame.draw.circle(bg, (255, 240, 170), (800, 90), 40)

    mask = pygame.Surface(size)
    mask.fill((0, 0, 0))
    white = (255, 255, 255)
    shapes = [
        lambda s, c: pygame.draw.rect(s, c, (0, h - 50, w, 50)),           # ground
        lambda s, c: pygame.draw.circle(s, c, (240, h - 50), 70),          # hill (slopes)
        lambda s, c: pygame.draw.rect(s, c, (420, h - 160, 160, 20)),      # floating platforms
        lambda s, c: pygame.draw.rect(s, c, (640, h - 260, 140, 20)),
        lambda s, c: pygame.draw.rect(s, c, (180, h - 300, 150, 20)),
        lambda s, c: pygame.draw.polygon(s, c, [(760, h - 50), (900, h - 50), (900, h - 130)]),
    ]
    for draw in shapes:
        draw(mask, white)                                # solid in the mask
        draw(bg, (70, 140, 80))                          # visible in the background
    return bg, mask


def load_scene():
    try:
        bg = pygame.image.load(BG_PATH)          
        mask_img = pygame.image.load(MASK_PATH)  
    except (FileNotFoundError, pygame.error) as e:
        print(f"Couldn't load '{BG_PATH}' / '{MASK_PATH}' ({e}) - using demo scene.")
        bg, mask_img = make_demo_scene()
    return bg, mask_img

def load_sprite():
    """Returns (right_facing, left_facing) surfaces, or None if no sprite file."""
    try:
        img = pygame.image.load(SPRITE_PATH).convert_alpha()
    except (FileNotFoundError, pygame.error):
        print(f"No sprite at '{SPRITE_PATH}' - using the yellow box.")
        return None
    img = img.subsurface(img.get_bounding_rect()).copy()      # trim transparent padding
    w = max(1, round(img.get_width() * SPRITE_HEIGHT / img.get_height()))
    img = pygame.transform.smoothscale(img, (w, SPRITE_HEIGHT))
    flipped = pygame.transform.flip(img, True, False)
    return (img, flipped) if SPRITE_FACES_RIGHT else (flipped, img)

def build_platform_mask(mask_img, size):
    """Turn any mask image into a pygame.Mask (set bits = solid)."""
    if mask_img.get_size() != size:
        mask_img = pygame.transform.scale(mask_img, size)

    flat = pygame.Surface(size)                          # 32-bit RGB
    flat.fill((0, 0, 0))
    flat.blit(mask_img, (0, 0))                          # transparency -> black

    rgb = pygame.surfarray.array3d(flat)                 # shape (w, h, 3)
    print("mask brightest value seen:", rgb.max(), flush=True)
    solid = rgb.max(axis=2) > 128                        # bright = solid

    alpha_surf = pygame.Surface(size, pygame.SRCALPHA)
    alpha_surf.fill((0, 0, 0, 0))
    pygame.surfarray.pixels_alpha(alpha_surf)[:] = solid * 255
    return pygame.mask.from_surface(alpha_surf, 127)


# ------------------------------------------------------------------ player ---
class Player:
    def __init__(self, platform, spawn, sprites=None):
        self.platform = platform
        self.sprites = sprites
        if sprites:
            self.w, self.h = sprites[0].get_size()
        else:
            self.w, self.h = FALLBACK_W, FALLBACK_H
        self.body = pygame.Mask((self.w, self.h), fill=True)
        self.spawn = spawn
        self.reset()

    def reset(self):
        self.x, self.y = self.spawn
        self.vx = self.vy = 0.0
        self.on_ground = False
        self.facing = 1
        for _ in range(200):                             # pop out if spawned inside terrain
            if not self.hits(self.x, self.y):
                break
            self.y -= 1

    def hits(self, x, y):
        return self.platform.overlap(self.body, (round(x), round(y))) is not None

    def jump(self):
        if self.on_ground:
            self.vy = -JUMP_SPEED

    def release_jump(self):
        if self.vy < 0:
            self.vy *= 0.5                               # variable jump height

    def update(self, direction):
        was_grounded = self.on_ground
        self.vx = direction * MOVE_SPEED
        if direction:
            self.facing = direction

        # --- horizontal, 1px at a time, with step-up for slopes and small ledges
        remaining = self.vx
        while abs(remaining) > 1e-6:
            step = max(-1.0, min(1.0, remaining))
            remaining -= step
            nx = self.x + step
            if not self.hits(nx, self.y):
                self.x = nx
                continue
            if was_grounded:
                for up in range(1, MAX_STEP + 1):
                    if not self.hits(nx, self.y - up):
                        self.x, self.y = nx, self.y - up
                        break
                else:
                    break
            else:
                break

        # keep on the downhill side of slopes instead of hopping off them
        if was_grounded and self.vy >= 0:
            start_y = self.y
            for _ in range(MAX_STEP + 1):
                if self.hits(self.x, self.y + 1):
                    break
                self.y += 1
            if not self.hits(self.x, self.y + 1):
                self.y = start_y                         # no ground nearby: let gravity take over

        # --- vertical
        self.vy = min(self.vy + GRAVITY, MAX_FALL)
        remaining = self.vy
        while abs(remaining) > 1e-6:
            step = max(-1.0, min(1.0, remaining))
            remaining -= step
            if self.hits(self.x, self.y + step):
                self.vy = 0.0
                break
            self.y += step

        self.on_ground = self.vy >= 0 and self.hits(self.x, self.y + 1)

    def clamp(self, w, h):
        self.x = max(0, min(self.x, w - self.w))
        if self.y > h + 100:
            self.reset()

    def draw(self, screen):
        if self.sprites:
            img = self.sprites[0] if self.facing == 1 else self.sprites[1]
            screen.blit(img, (round(self.x), round(self.y)))
            return
        r = pygame.Rect(round(self.x), round(self.y), self.w, self.h)
        pygame.draw.rect(screen, (30, 30, 60), r.inflate(2, 2), border_radius=5)
        pygame.draw.rect(screen, (250, 200, 60), r, border_radius=4)
        eye_x = r.centerx + 3 * self.facing
        pygame.draw.circle(screen, (20, 20, 20), (eye_x, r.top + 10), 3)


# -------------------------------------------------------------------- main ---
def find_spawn(platform, w, h, pw, ph):
    cx = int(platform.centroid()[0])
    for x in sorted(range(w), key=lambda v: abs(v - cx)):
        for y in range(h):
            if platform.get_at((x, y)):
                return (x - pw // 2, max(0, y - ph - 5))
    return (w // 2 - pw // 2, 0)


def main():
    pygame.init()
    bg, mask_img = load_scene()

    dw, dh = pygame.display.get_desktop_sizes()[0]
    scale = min(1.0, (dw * 0.9) / bg.get_width(), (dh * 0.85) / bg.get_height())
    size = (int(bg.get_width() * scale), int(bg.get_height() * scale))

    screen = pygame.display.set_mode(size)
    bg = bg.convert()
    if bg.get_size() != size:
        bg = pygame.transform.smoothscale(bg, size)
    mask_img = mask_img.convert_alpha()
    clock = pygame.time.Clock()

    platform = build_platform_mask(mask_img, size)
    solid = platform.count()
    print(f"solid pixels: {solid} of {size[0] * size[1]}", flush=True)
    pygame.display.set_caption(f"Platformer - solid {solid}/{size[0] * size[1]}")

    overlay = platform.to_surface(setcolor=MASK_TINT, unsetcolor=(0, 0, 0, 0)).convert_alpha()
    show_mask = SHOW_MASK
    sprites = load_sprite()
    pw, ph = sprites[0].get_size() if sprites else (FALLBACK_W, FALLBACK_H)
    player = Player(platform, spawn=find_spawn(platform, *size, pw, ph), sprites=sprites)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                    player.jump()
                elif event.key == pygame.K_m:
                    show_mask = not show_mask
            elif event.type == pygame.KEYUP:
                if event.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                    player.release_jump()

        keys = pygame.key.get_pressed()
        direction = (keys[pygame.K_RIGHT] or keys[pygame.K_d]) - (keys[pygame.K_LEFT] or keys[pygame.K_a])

        player.update(direction)
        player.clamp(*size)

        screen.blit(bg, (0, 0))
        if show_mask:
            screen.blit(overlay, (0, 0))
        player.draw(screen)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()