import asyncio
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import os
from playwright.async_api import async_playwright, Page

try:
    from loguru import logger  # type: ignore
except Exception:  # pragma: no cover
    class _SimpleLogger:
        def info(self, msg: str):
            print(f"[INFO] {msg}")

        def warning(self, msg: str):
            print(f"[WARN] {msg}")

        def success(self, msg: str):
            print(f"[OK] {msg}")

        def error(self, msg: str):
            print(f"[ERR] {msg}")

    logger = _SimpleLogger()  # type: ignore

# Optional scientific deps
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # type: ignore
    from io import BytesIO  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    BytesIO = None  # type: ignore

# Optional imitation imports
try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    import joblib  # type: ignore
except Exception:  # pragma: no cover
    LogisticRegression = None
    joblib = None

GAME_URL = "https://snake.io/"
STORAGE_STATE_PATH = os.path.abspath(os.path.join("data", "storage_state.json"))


@dataclass
class PolicyParams:
    food_threshold: int = 220
    obstacle_threshold: int = 80
    boost_gain: float = 1.2
    border_margin_px: int = 80
    center_bias: float = 0.15


# ---- Image utils (with graceful fallbacks) ----

def decode_frame_bgr(buf: bytes):
    if cv2 is not None and np is not None:
        try:
            arr = np.frombuffer(buf, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img
        except Exception:
            pass
    if Image is not None and BytesIO is not None:
        try:
            im = Image.open(BytesIO(buf)).convert("RGB")
            rgb = list(im.getdata())
            w, h = im.size
            # convert to BGR nested list HxWx3
            bgr = [[(p[2], p[1], p[0]) for p in rgb[i*w:(i+1)*w]] for i in range(h)]
            return bgr
        except Exception:
            return None
    return None


def bgr_to_gray(frame_bgr):
    if cv2 is not None and np is not None:
        try:
            return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            pass
    # frame_bgr may be list when PIL path used
    if np is None:
        # manual conversion to 2D list of uint8
        h = len(frame_bgr)
        w = len(frame_bgr[0]) if h > 0 else 0
        gray = [[0] * w for _ in range(h)]
        for y in range(h):
            row = frame_bgr[y]
            for x in range(w):
                b, g, r = row[x]
                v = int(0.114 * b + 0.587 * g + 0.299 * r)
                gray[y][x] = v
        return gray
    else:
        b = frame_bgr[..., 0].astype("float32")
        g = frame_bgr[..., 1].astype("float32")
        r = frame_bgr[..., 2].astype("float32")
        gray = 0.114 * b + 0.587 * g + 0.299 * r
        return gray.astype("uint8")


def resize_gray(gray, size: Tuple[int, int]):
    w, h = size
    if cv2 is not None and np is not None:
        try:
            return cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    if Image is not None and BytesIO is not None and np is not None:
        try:
            im = Image.fromarray(gray)
            im = im.resize((w, h), Image.BILINEAR)
            return np.asarray(im)
        except Exception:
            pass
    # pure python naive downscale
    if np is None:
        H = len(gray)
        W = len(gray[0]) if H > 0 else 0
        ys = [int(i * (H - 1) / max(1, h - 1)) for i in range(h)]
        xs = [int(i * (W - 1) / max(1, w - 1)) for i in range(w)]
        out = [[gray[ys[yy]][xs[xx]] for xx in range(w)] for yy in range(h)]
        return out
    else:
        H, W = gray.shape[:2]
        ys = (np.linspace(0, H - 1, h)).astype(int)
        xs = (np.linspace(0, W - 1, w)).astype(int)
        return gray[np.ix_(ys, xs)]


def simple_edges(gray):
    if cv2 is not None and np is not None:
        try:
            return cv2.Canny(gray, 50, 150)
        except Exception:
            pass
    if np is None:
        H = len(gray)
        W = len(gray[0]) if H > 0 else 0
        edges = [[0] * W for _ in range(H)]
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                gx = (gray[y][x + 1] - gray[y][x - 1]) * 0.5
                gy = (gray[y + 1][x] - gray[y - 1][x]) * 0.5
                mag = (gx * gx + gy * gy) ** 0.5
                edges[y][x] = 255 if mag > 30 else 0
        return edges
    else:
        grayf = gray.astype("float32") / 255.0
        gx = np.zeros_like(grayf)
        gy = np.zeros_like(grayf)
        gx[:, 1:-1] = (grayf[:, 2:] - grayf[:, :-2]) * 0.5
        gy[1:-1, :] = (grayf[2:, :] - grayf[:-2, :]) * 0.5
        mag = (gx * gx + gy * gy) ** 0.5
        return (mag > 0.12).astype("uint8") * 255


# ---- Policy ----

def extract_simple_features(frame_bgr):
    if np is None:
        # minimal feature vector for fallback
        return [0.0] * (32 * 18)
    gray = bgr_to_gray(frame_bgr)
    small = resize_gray(gray, (32, 18))
    feats = (small.astype("float32") / 255.0).flatten()
    return feats


def compute_direction_scores(frame_bgr, params: PolicyParams):
    if np is None:
        # fallback: bias to right, then down, then left, then up
        return [1.0, 0.2, 0.1, 0.1], 0
    h, w = frame_bgr.shape[:2]
    gray = bgr_to_gray(frame_bgr)

    margin = params.border_margin_px
    safe_mask = (np.zeros_like(gray, dtype="uint8"))
    safe_mask[margin : h - margin, margin : w - margin] = 1

    food_mask = (gray > params.food_threshold).astype("uint8") * safe_mask
    edges = simple_edges(gray)
    obstacle_mask = ((gray < params.obstacle_threshold) | (edges > 0)).astype("uint8")

    cx, cy = w // 2, h // 2
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    ray_len = int(min(h, w) * 0.45)
    scores: List[float] = []
    for dx, dy in directions:
        food_score = 0.0
        danger_score = 0.0
        for r in range(10, ray_len, 8):
            x = int(max(0, min(w - 1, cx + dx * r)))
            y = int(max(0, min(h - 1, cy + dy * r)))
            food_score += float(food_mask[y, x])
            danger_score += float(obstacle_mask[y, x]) * (1 + r / max(1, ray_len))
        center_penalty = params.center_bias * (abs(dx * 40) + abs(dy * 40))
        scores.append(food_score - 2.0 * danger_score - center_penalty)
    scores_arr = np.asarray(scores, dtype="float32")
    best_dir_idx = int(scores_arr.argmax())
    return scores_arr, best_dir_idx


def choose_direction(frame_bgr, params: PolicyParams, explore_prob: float):
    if np is None or frame_bgr is None:
        # fallback oscillation
        t = int(time.time() * 2) % 4
        use_boost = (t % 2) == 0
        scores = [1.0, 0.8, 0.6, 0.4]
        return t, use_boost, scores, "fallback"
    scores, best_idx = compute_direction_scores(frame_bgr, params)
    mode = "exploit"
    if np.random.random() < explore_prob:
        top2 = np.argsort(-scores)[:2]
        best_idx = int(np.random.choice(top2))
        mode = "explore"
    use_boost = scores[best_idx] > params.boost_gain * float(np.mean(scores))
    return best_idx, bool(use_boost), scores, mode


async def any_play_visible(page: Page) -> bool:
    selectors = [
        "role=button[name=/play again/i]",
        "role=button[name=/play now/i]",
        "role=button[name=/play/i]",
        "text=/play again/i]",
        "text=/play now/i]",
        "text=/play/i",
    ]
    for sel in selectors:
        try:
            if await page.locator(sel).first.is_visible():
                return True
        except Exception:
            continue
    return False


async def try_click_selectors(page: Page, selectors: List[str], timeout_ms: int = 0) -> bool:
    for sel in selectors:
        try:
            loc = page.locator(sel).first
            if await loc.is_visible():
                await loc.click(timeout=timeout_ms or 0)
                return True
        except Exception:
            continue
    return False


async def ensure_cookie_consent(page: Page) -> None:
    try:
        accept_selectors = [
            "role=button[name=/accept/i]",
            "role=button[name=/i agree/i]",
            "role=button[name=/consent/i]",
            "text=/accept all/i",
            "text=/accept/i",
            "text=/i agree/i",
            "text=/got it/i",
        ]
        if await try_click_selectors(page, accept_selectors):
            logger.info("Cookie consent accepted (top-level)")
            return
        for frame in page.frames:
            try:
                if frame == page.main_frame:
                    continue
                if await try_click_selectors(frame, accept_selectors):  # type: ignore[arg-type]
                    logger.info("Cookie consent accepted (iframe)")
                    return
            except Exception:
                continue
    except Exception:
        pass


async def try_close_popups(page: Page) -> None:
    selectors = [
        "role=button[name=/close/i]",
        "text=/close/i",
        "text=/skip/i",
        "text=/continue/i",
        "text=/ok/i",
        "text=/dismiss/i",
        "text=/no thanks/i",
        "[aria-label=\"close\"]",
        "button:has-text('X')",
    ]
    try:
        await try_click_selectors(page, selectors)
    except Exception:
        pass


async def get_game_center(page: Page) -> tuple[int, int, int]:
    try:
        box = await page.evaluate(
            """
            () => {
              const canvases = Array.from(document.querySelectorAll('canvas'));
              let best = null;
              let bestArea = 0;
              for (const c of canvases) {
                const r = c.getBoundingClientRect();
                const area = Math.max(0, r.width) * Math.max(0, r.height);
                if (area > bestArea && r.width > 100 && r.height > 100) { bestArea = area; best = r; }
              }
              if (best) return { x: best.left + best.width/2, y: best.top + best.height/2, w: best.width, h: best.height };
              const r = document.documentElement.getBoundingClientRect();
              return { x: r.width/2, y: r.height/2, w: Math.min(1024, r.width), h: Math.min(768, r.height) };
            }
            """
        )
        cx = int(float(box["x"]))
        cy = int(float(box["y"]))
        radius = int(0.45 * float(min(box["w"], box["h"])) )
        return cx, cy, max(radius, 60)
    except Exception:
        vp = page.viewport_size or {"width": 1024, "height": 768}
        cx, cy = vp["width"] // 2, vp["height"] // 2
        radius = int(min(vp["width"], vp["height"]) * 0.4)
        return cx, cy, max(radius, 60)


async def press_for_direction(page: Page, dir_idx: int, boost: bool) -> None:
    cx, cy, radius = await get_game_center(page)
    vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    dx, dy = vectors[dir_idx]
    target_x = int(cx + dx * radius)
    target_y = int(cy + dy * radius)
    await page.mouse.move(target_x, target_y)
    if boost:
        await page.mouse.down(button="left")
    else:
        await page.mouse.up(button="left")
    await page.wait_for_timeout(20)


async def ensure_started(page: Page, mode: str = "auto", max_clicks: int = 3) -> None:
    mode = (mode or "auto").lower()
    try:
        await page.bring_to_front()
    except Exception:
        pass
    await try_close_popups(page)
    await ensure_cookie_consent(page)

    if mode == "manual":
        deadline = time.time() + 120
        while time.time() < deadline:
            try:
                if not await any_play_visible(page):
                    return
            except Exception:
                return
            await page.wait_for_timeout(300)
        return

    played = False
    for _ in range(max_clicks):
        try:
            clicked = await try_click_selectors(
                page,
                [
                    "role=button[name=/play again/i]",
                    "role=button[name=/play now/i]",
                    "role=button[name=/play/i]",
                    "text=/play again/i]",
                    "text=/play now/i]",
                    "text=/play/i",
                ],
            )
            if clicked:
                played = True
                await page.wait_for_timeout(700)
                break
        except Exception:
            continue
        await page.wait_for_timeout(350)

    try:
        cx, cy, _ = await get_game_center(page)
        await page.mouse.click(cx, cy)
        await page.wait_for_timeout(400)
        played = True
    except Exception:
        pass

    if played:
        logger.info("Game start attempted")


async def is_game_over(page: Page) -> bool:
    try:
        return await any_play_visible(page)
    except Exception:
        return False


def mutate(params: PolicyParams, scale: float = 0.1) -> PolicyParams:
    if np is None:
        # basic fallback: small bounded tweaks using time
        import random
        random.seed(time.time())
        return PolicyParams(
            food_threshold=min(255, max(100, params.food_threshold + random.randint(-2, 2))),
            obstacle_threshold=min(200, max(10, params.obstacle_threshold + random.randint(-2, 2))),
            boost_gain=float(min(3.0, max(0.5, params.boost_gain + (random.random() - 0.5) * 0.02))),
            border_margin_px=min(200, max(10, params.border_margin_px + random.randint(-3, 3))),
            center_bias=float(min(1.5, max(0.0, params.center_bias + (random.random() - 0.5) * 0.02))),
        )
    rng = np.random.default_rng()
    def j(x, low, high):
        return int(np.clip(x, low, high))
    return PolicyParams(
        food_threshold=j(params.food_threshold + rng.normal(0, 20 * scale), 100, 255),
        obstacle_threshold=j(params.obstacle_threshold + rng.normal(0, 15 * scale), 10, 200),
        boost_gain=float(np.clip(params.boost_gain + rng.normal(0, 0.2 * scale), 0.5, 3.0)),
        border_margin_px=j(params.border_margin_px + rng.normal(0, 30 * scale), 10, 200),
        center_bias=float(np.clip(params.center_bias + rng.normal(0, 0.2 * scale), 0.0, 1.5)),
    )


async def play_session(
    params: PolicyParams,
    episodes: int,
    seconds_cap: int = 120,
    headless: bool = True,
    model_path: Optional[str] = None,
    warmup_seconds: int = 8,
    optimize: bool = True,
    auto_respawn: bool = True,
    start_mode: str = "auto",
) -> List[float]:
    round_scores: List[float] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        storage_state_arg = STORAGE_STATE_PATH if os.path.exists(STORAGE_STATE_PATH) else None
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            storage_state=storage_state_arg,
        )
        page = await ctx.new_page()
        await page.goto(GAME_URL, wait_until="domcontentloaded")

        try:
            await ensure_cookie_consent(page)
            await ctx.storage_state(path=STORAGE_STATE_PATH)
        except Exception:
            pass

        if warmup_seconds > 0:
            logger.info(f"Warmup: waiting {warmup_seconds}s...")
            await page.wait_for_timeout(warmup_seconds * 1000)

        clf = None
        if model_path and os.path.exists(model_path) and LogisticRegression is not None and np is not None:
            try:
                import joblib as _joblib
                clf = _joblib.load(model_path)
                logger.info(f"Loaded imitation model from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")

        try:
            await page.add_init_script(
                """
                (function(){
                  if (window.__botOverlay) return;
                  const d = document.createElement('div');
                  d.id = '__botOverlay';
                  d.style.position = 'fixed';
                  d.style.top = '8px';
                  d.style.left = '8px';
                  d.style.zIndex = '999999';
                  d.style.padding = '6px 8px';
                  d.style.fontFamily = 'monospace';
                  d.style.fontSize = '12px';
                  d.style.color = '#00FF88';
                  d.style.background = 'rgba(0,0,0,0.5)';
                  d.style.border = '1px solid rgba(0,255,136,0.5)';
                  d.textContent = 'BOT: initializing...';
                  document.body.appendChild(d);
                  window.__botOverlay = d;
                })();
                """
            )
        except Exception:
            pass

        await ensure_started(page, mode=start_mode)
        await page.wait_for_timeout(400)

        best_params = params
        best_score = -1e9

        ep = 0
        explore_prob = 0.6
        def should_continue() -> bool:
            return episodes < 0 or ep < episodes

        while should_continue():
            start = time.time()
            last_score = 0.0
            while True:
                if time.time() - start > seconds_cap:
                    break
                if (time.time() - start) > 3.0 and await is_game_over(page):
                    break

                try:
                    box = await page.evaluate(
                        """() => { const c=document.querySelector('canvas'); if(!c){return null;} const r=c.getBoundingClientRect(); return {x:r.left,y:r.top,w:r.width,h:r.height}; }"""
                    )
                except Exception:
                    box = None
                if box and box.get("w", 0) > 100 and box.get("h", 0) > 100:
                    clip = {
                        "x": max(0, float(box["x"])),
                        "y": max(0, float(box["y"])),
                        "width": float(box["w"]),
                        "height": float(box["h"]),
                    }
                    buf = await page.screenshot(clip=clip)
                else:
                    buf = await page.screenshot(full_page=False)

                frame = decode_frame_bgr(buf)

                mode = "heuristic" if np is not None else "fallback"
                if clf is not None and np is not None and np.random.random() > explore_prob and frame is not None:
                    feats = extract_simple_features(frame)
                    feats_arr = np.asarray(feats, dtype="float32").reshape(1, -1)
                    dir_idx = int(clf.predict(feats_arr)[0])
                    _, boost, _, _ = choose_direction(frame, params, 0.0)
                    mode = "imitate"
                else:
                    dir_idx, boost, scores, ex_mode = choose_direction(frame, params, explore_prob)
                    mode = ex_mode

                await press_for_direction(page, dir_idx, boost)
                try:
                    cx, cy, r = await get_game_center(page)
                    await page.mouse.move(cx + int(0.05*r), cy)
                    await page.mouse.move(cx - int(0.05*r), cy)
                except Exception:
                    pass

                if frame is not None:
                    if np is None:
                        last_score = (time.time() - start)
                    else:
                        gray = bgr_to_gray(frame)
                        last_score = (time.time() - start) + 0.0005 * float(np.sum(gray > params.food_threshold))
                else:
                    last_score = (time.time() - start)

                try:
                    elapsed = time.time() - start
                    overlay_text = (
                        f"BOT mode={mode} ep={ep+1 if episodes>0 else ep+1} t={elapsed:0.1f}s\n"
                        f"score={last_score:0.2f} explore={explore_prob:0.2f}\n"
                        f"params: food_thr={params.food_threshold} obst_thr={params.obstacle_threshold} border={params.border_margin_px} center={params.center_bias:0.2f}"
                    )
                    await page.evaluate("(t)=>{ if(window.__botOverlay) window.__botOverlay.textContent = t; }", overlay_text)
                except Exception:
                    pass

            round_scores.append(last_score)
            logger.info(f"Round ended ep={ep+1 if episodes>0 else ep+1}/{'∞' if episodes<0 else episodes} score={last_score:.2f}")

            if optimize and last_score > best_score:
                best_params = params
                best_score = last_score
            if optimize:
                if last_score < 0.7 * max(1.0, best_score):
                    params.border_margin_px = min(200, max(10, params.border_margin_px + 5))
                    params.center_bias = float(min(1.5, max(0.0, params.center_bias + 0.02)))
                else:
                    params.center_bias = float(min(1.5, max(0.0, params.center_bias - 0.01)))
                params = mutate(best_params, scale=0.20)
                if np is not None:
                    explore_prob = max(0.05, explore_prob * 0.97)

            if await is_game_over(page):
                if auto_respawn:
                    wait_deadline = time.time() + 8.0
                    while time.time() < wait_deadline and await is_game_over(page):
                        await page.wait_for_timeout(250)
                if await is_game_over(page):
                    await ensure_started(page, mode=start_mode)
                    await page.wait_for_timeout(600)

            await page.wait_for_timeout(800)
            ep += 1

        await browser.close()

    return round_scores


async def demo_and_train(model_path: str, demo_seconds: int) -> None:
    if demo_seconds <= 0:
        return
    if LogisticRegression is None or np is None:
        logger.warning("Training requires scikit-learn and numpy; skipping")
        return
    os.makedirs("data", exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        ctx = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await ctx.new_page()
        await page.goto(GAME_URL, wait_until="domcontentloaded")
        await ensure_started(page, mode="auto")
        await page.wait_for_timeout(1000)

        await page.add_init_script(
            """
            window.__keysDown = new Set();
            window.__mouseX = 0; window.__mouseY = 0; window.__mouseDown = false;
            window.addEventListener('keydown', e => window.__keysDown.add(e.code));
            window.addEventListener('keyup', e => window.__keysDown.delete(e.code));
            window.addEventListener('mousemove', e => { window.__mouseX = e.clientX; window.__mouseY = e.clientY; });
            window.addEventListener('mousedown', e => { if (e.button === 0) window.__mouseDown = true; });
            window.addEventListener('mouseup', e => { if (e.button === 0) window.__mouseDown = false; });
            """
        )

        start = time.time()
        X_list: List = []
        y_list: List[int] = []
        while time.time() - start < demo_seconds:
            buf = await page.screenshot(full_page=False)
            frame = decode_frame_bgr(buf)
            if frame is None:
                continue
            feats = extract_simple_features(frame)
            feats = np.asarray(feats, dtype="float32") if np is not None else None
            vp = page.viewport_size or {"width": 1280, "height": 800}
            center_x, center_y = vp["width"] // 2, vp["height"] // 2
            state = await page.evaluate(
                "() => ({ x: window.__mouseX, y: window.__mouseY, md: window.__mouseDown, keys: Array.from(window.__keysDown) })"
            )
            dx, dy = state["x"] - center_x, state["y"] - center_y
            dir_idx = None
            if abs(dx) + abs(dy) > 10:
                if abs(dx) >= abs(dy):
                    dir_idx = 0 if dx >= 0 else 2
                else:
                    dir_idx = 1 if dy >= 0 else 3
            else:
                k = set(state["keys"]) if state and state.get("keys") else set()
                if {"KeyD", "ArrowRight"} & k:
                    dir_idx = 0
                elif {"KeyS", "ArrowDown"} & k:
                    dir_idx = 1
                elif {"KeyA", "ArrowLeft"} & k:
                    dir_idx = 2
                elif {"KeyW", "ArrowUp"} & k:
                    dir_idx = 3
            if dir_idx is not None and feats is not None:
                X_list.append(feats)
                y_list.append(dir_idx)
            await page.wait_for_timeout(50)

        await browser.close()
        if np is None:
            logger.warning("NumPy missing; cannot save demos")
            return
        if len(X_list) >= 50:
            X = np.stack(X_list)
            y = np.array(y_list, dtype=np.int64)
            np.savez("data/demos.npz", X=X, y=y)
            logger.success(f"Saved demos: {X.shape} -> data/demos.npz")
        else:
            logger.warning("Too few demo samples collected; skipping save")

    try:
        data = np.load("data/demos.npz")
        X, y = data["X"], data["y"]
        clf = LogisticRegression(max_iter=1000, multi_class="auto")
        clf.fit(X, y)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import joblib as _joblib
        _joblib.dump(clf, model_path)
        acc = float(clf.score(X, y))
        logger.success(f"Trained imitation model acc={acc:.3f} -> {model_path}")
    except Exception as e:
        logger.error(f"Training failed: {e}")


async def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=45)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--demo-seconds", type=int, default=0)
    parser.add_argument("--train-after-demo", action="store_true")
    parser.add_argument("--model-path", type=str, default=os.path.join("data", "model.joblib"))
    parser.add_argument("--warmup-seconds", type=int, default=10)
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--no-auto-respawn", action="store_true")
    parser.add_argument("--infinite", action="store_true")
    parser.add_argument("--start-mode", type=str, default="auto", choices=["auto", "manual"]) 
    args = parser.parse_args()

    if args.demo_seconds > 0:
        await demo_and_train(args.model_path, args.demo_seconds)
        if args.train_after_demo and args.episodes <= 0:
            return

    model_to_use = args.model_path if os.path.exists(args.model_path) else None
    initial_params = PolicyParams()
    target_episodes = -1 if args.infinite else max(1, args.episodes)
    scores = await play_session(
        initial_params,
        episodes=target_episodes,
        seconds_cap=args.seconds,
        headless=args.headless,
        model_path=model_to_use,
        warmup_seconds=args.warmup_seconds,
        optimize=args.optimize,
        auto_respawn=not args.no_auto_respawn,
        start_mode=args.start_mode,
    )
    if len(scores) > 0:
        try:
            if np is not None:
                import numpy as _np
                avg = float(_np.mean(_np.asarray(scores)))
            else:
                avg = sum(scores) / len(scores)
        except Exception:
            avg = sum(scores) / len(scores)
        logger.info(f"Finished session. Rounds played: {len(scores)}; best score={max(scores):.2f}; avg={avg:.2f}")


if __name__ == "__main__":
    asyncio.run(main())