"""
6th Tool web scraper utilities.
Extracts all match CSVs and ingests them into the configured GT database with dedupe.
"""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import os
import tempfile
import time

import pandas as pd

from db_manager import GTBaseballDB


def _is_probably_html(raw_bytes: bytes) -> bool:
    preview = raw_bytes[:4096].decode("utf-8", errors="ignore").lower()
    return "<html" in preview or "<!doctype html" in preview or "<body" in preview


def _switch_to_context_with_game_controls(driver, by) -> bool:
    """Switch into default content or an iframe likely containing game controls."""
    try:
        driver.switch_to.default_content()
    except Exception:
        pass

    frames = [None]
    try:
        frames.extend(driver.find_elements(by.TAG_NAME, "iframe"))
    except Exception:
        pass

    for frame in frames:
        try:
            if frame is None:
                driver.switch_to.default_content()
            else:
                driver.switch_to.default_content()
                driver.switch_to.frame(frame)

            selects = driver.find_elements(by.TAG_NAME, "select")
            for sel in selects:
                if not sel.is_displayed():
                    continue
                if len(sel.find_elements(by.TAG_NAME, "option")) >= 2:
                    return True

            if driver.find_elements(by.ID, "game-dl-pbp-csv"):
                return True
        except Exception:
            continue

    try:
        driver.switch_to.default_content()
    except Exception:
        pass
    return False


def _pick_game_dropdown(driver, by):
    """Choose the best candidate dropdown from visible <select> controls."""
    select_nodes = driver.find_elements(by.TAG_NAME, "select")
    ranked = []
    for node in select_nodes:
        try:
            if not node.is_displayed():
                continue
            opts = node.find_elements(by.TAG_NAME, "option")
            if len(opts) < 2:
                continue
            ident = f"{node.get_attribute('id') or ''} {node.get_attribute('name') or ''} {node.get_attribute('class') or ''}".lower()
            score = 0
            if "game" in ident or "match" in ident:
                score += 2
            loc = node.location
            if loc.get("x", 9999) < 550 and loc.get("y", 9999) < 380:
                score += 1
            ranked.append((score, loc.get("y", 9999), loc.get("x", 9999), node))
        except Exception:
            continue
    if not ranked:
        return None
    ranked.sort(key=lambda x: (-x[0], x[1], x[2]))
    return ranked[0][3]


def scrape_all_matches_to_db(
    username: str,
    password: str,
    base_url: str = "https://6thtool.smt.com",
) -> dict:
    """Log in, iterate all match options, download CSVs, and ingest into DB with dedupe."""
    try:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.keys import Keys
        from selenium.webdriver.support.ui import WebDriverWait, Select
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except Exception as exc:
        raise RuntimeError(
            "Missing scraper dependencies. Install with: pip install selenium webdriver-manager"
        ) from exc

    if not username or not password:
        raise RuntimeError("Username and password are required.")

    # Use the configured DB target. If Turso credentials are present, this writes there.
    db = GTBaseballDB("data/gt_baseball.db")
    run_summary = {
        "inserted": 0,
        "skipped": 0,
        "duplicate": 0,
        "errors": 0,
        "attempted_matches": 0,
        "successful_matches": 0,
    }

    fatal_error = None
    try:
        with tempfile.TemporaryDirectory(prefix="sixthtool_dbsync_") as download_dir:
            options = webdriver.ChromeOptions()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1600,1000")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_experimental_option(
                "prefs",
                {
                    "download.default_directory": download_dir,
                    "download.prompt_for_download": False,
                    "download.directory_upgrade": True,
                    "safebrowsing.enabled": True,
                },
            )

            driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
            wait = WebDriverWait(driver, 30)
            try:
                try:
                    driver.execute_cdp_cmd(
                        "Page.setDownloadBehavior",
                        {"behavior": "allow", "downloadPath": download_dir},
                    )
                except Exception:
                    pass

                driver.get(base_url.rstrip("/"))
                user_el = wait.until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            "//input[@type='email' or contains(translate(@name,'EMAILUSERLOGIN','emailuserlogin'),'email') or contains(translate(@name,'EMAILUSERLOGIN','emailuserlogin'),'user') or @type='text']",
                        )
                    )
                )
                pass_el = wait.until(EC.presence_of_element_located((By.XPATH, "//input[@type='password']")))
                user_el.clear()
                user_el.send_keys(username)
                pass_el.clear()
                pass_el.send_keys(password)
                pass_el.send_keys(Keys.ENTER)

                wait.until(lambda d: len(d.find_elements(By.XPATH, "//button|//a|//select")) > 0)
                time.sleep(1.5)
                _switch_to_context_with_game_controls(driver, By)

                dropdown = _pick_game_dropdown(driver, By)
                if dropdown is None:
                    raise RuntimeError("No match dropdown found after login.")

                selector = Select(dropdown)
                match_names = []
                for opt in selector.options:
                    label = (opt.text or "").strip()
                    if not label:
                        continue
                    low = label.lower()
                    if low in ("select", "select game", "select match"):
                        continue
                    match_names.append(label)
                match_names = list(dict.fromkeys(match_names))
                if not match_names:
                    raise RuntimeError("Match dropdown found, but no match options were available.")

                for match_name in match_names:
                    run_summary["attempted_matches"] += 1
                    try:
                        _switch_to_context_with_game_controls(driver, By)
                        dropdown = _pick_game_dropdown(driver, By)
                        if dropdown is None:
                            raise RuntimeError("Dropdown disappeared while processing matches.")

                        selector = Select(dropdown)
                        selected_ok = False
                        try:
                            selector.select_by_visible_text(match_name)
                            selected_ok = True
                        except Exception:
                            for opt in selector.options:
                                t = (opt.text or "").strip()
                                if t and match_name.lower() in t.lower():
                                    selector.select_by_visible_text(t)
                                    selected_ok = True
                                    break
                        if not selected_ok:
                            raise RuntimeError(f"Could not select match '{match_name}'.")

                        try:
                            driver.execute_script(
                                "arguments[0].dispatchEvent(new Event('change', {bubbles: true}));",
                                dropdown,
                            )
                        except Exception:
                            pass
                        time.sleep(1.0)

                        try:
                            btn = wait.until(EC.presence_of_element_located((By.ID, "game-dl-pbp-csv")))
                        except Exception:
                            btn = wait.until(
                                EC.element_to_be_clickable(
                                    (By.XPATH, "//button[contains(normalize-space(.),'Download Derived CSV')] | //a[contains(normalize-space(.),'Download Derived CSV')]")
                                )
                            )
                        pre_files = set(os.listdir(download_dir))
                        try:
                            btn.click()
                        except Exception:
                            driver.execute_script("arguments[0].click();", btn)

                        latest_path = None
                        start = time.time()
                        while time.time() - start < 45:
                            files = set(os.listdir(download_dir))
                            new_files = files - pre_files
                            complete_files = [
                                os.path.join(download_dir, f)
                                for f in new_files
                                if not f.lower().endswith(".crdownload") and not f.lower().endswith(".part")
                            ]
                            if complete_files:
                                latest_path = max(complete_files, key=os.path.getmtime)
                                break
                            time.sleep(0.5)
                        if latest_path is None:
                            raise RuntimeError("No file downloaded for this match.")

                        with open(latest_path, "rb") as fh:
                            raw_bytes = fh.read()
                        if not raw_bytes or _is_probably_html(raw_bytes):
                            raise RuntimeError("Downloaded payload is not CSV data.")

                        df = pd.read_csv(io.BytesIO(raw_bytes))
                        game_label = match_name
                        # Keep the original downloaded CSV filename so DB duplicate checks
                        # compare against the real 6th Tool artifact name already stored.
                        file_name = os.path.basename(latest_path) or f"{match_name}.csv"
                        result = db.ingest_dataframe(
                            df,
                            game_label=game_label,
                            file_name=file_name,
                            skip_if_exists=True,
                        )
                        status = result.get("status", "skipped")
                        if status == "inserted":
                            run_summary["inserted"] += 1
                        elif status == "duplicate":
                            run_summary["duplicate"] += 1
                        else:
                            run_summary["skipped"] += 1
                        run_summary["successful_matches"] += 1
                    except Exception:
                        run_summary["errors"] += 1
                        continue

            finally:
                driver.quit()
    except Exception as exc:
        run_summary["errors"] += 1
        fatal_error = exc
    finally:
        now_utc = datetime.now(timezone.utc).isoformat()
        db.set_meta("last_scrape_run_at", now_utc)
        db.set_meta("last_scrape_summary", json.dumps(run_summary))

    if fatal_error is not None:
        raise fatal_error
    return run_summary
