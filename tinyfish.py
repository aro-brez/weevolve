#!/usr/bin/env python3
"""
TinyFish Web Agent API Client for WeEvolve
==========================================
Enterprise-grade web automation for competitive intelligence and bookmark processing.

Features:
  - Server-Sent Events (SSE) streaming for real-time progress
  - Parallel batch scanning with rate limiting
  - Anti-detection (stealth mode + residential proxies)
  - Cost tracking integration
  - Retry logic with exponential backoff

Usage:
  from weevolve.tinyfish import TinyFishClient

  client = TinyFishClient()
  result = client.scan_url("https://example.com", "Extract pricing")

  # Batch scanning
  results = client.scan_batch(urls, "Categorize as TOOLS/COMPETITORS/UI_UX")

API Reference: https://docs.mino.ai/
Pricing: $0.015/step (Pay As You Go) or $15/mo (Standard, 1650 steps)

(C) LIVE FREE = LIVE FOREVER
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import concurrent.futures

# WeEvolve config imports
from weevolve.config import DATA_DIR, COST_LOG

# Constants
TINYFISH_API_BASE = "https://agent.tinyfish.ai/v1"
TINYFISH_API_ENDPOINT = f"{TINYFISH_API_BASE}/automation/run-sse"
DEFAULT_TIMEOUT = 120  # 2 minutes per scan
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
COST_PER_STEP = 0.015  # Pay As You Go pricing


@dataclass
class TinyFishResult:
    """Result from a TinyFish scan."""
    url: str
    goal: str
    status: str  # COMPLETED, FAILED, TIMEOUT
    result_json: Optional[Dict] = None
    error: Optional[str] = None
    steps: int = 0
    cost: float = 0.0
    duration: float = 0.0
    timestamp: str = ""


class TinyFishClient:
    """TinyFish Web Agent API client with SSE streaming."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize TinyFish client.

        Args:
            api_key: TinyFish API key (defaults to TINYFISH_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("TINYFISH_API_KEY")
        if not self.api_key:
            raise ValueError(
                "TinyFish API key not found. Set TINYFISH_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self.session = requests.Session()
        self.session.headers.update({
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        })

    def scan_url(
        self,
        url: str,
        goal: str,
        stealth: bool = True,
        timeout: int = DEFAULT_TIMEOUT,
        retry: bool = True
    ) -> TinyFishResult:
        """Scan a URL and extract structured data.

        Args:
            url: Target URL to scan
            goal: Natural language description of what to extract
            stealth: Use stealth mode (anti-detection, slower but more reliable)
            timeout: Maximum time to wait for completion (seconds)
            retry: Retry on failure

        Returns:
            TinyFishResult with extracted data or error

        Example:
            result = client.scan_url(
                "https://openclaw.com/pricing",
                "Extract all pricing tiers and features"
            )
            if result.status == "COMPLETED":
                print(result.result_json)
        """
        start_time = time.time()

        payload = {
            "url": url,
            "goal": goal,
            "browser_profile": "stealth" if stealth else "lite"
        }

        attempts = 0
        max_attempts = MAX_RETRIES if retry else 1

        while attempts < max_attempts:
            attempts += 1

            try:
                response = self.session.post(
                    TINYFISH_API_ENDPOINT,
                    json=payload,
                    stream=True,
                    timeout=timeout
                )
                response.raise_for_status()

                # Parse SSE stream
                result = self._parse_sse_stream(response, url, goal)
                result.duration = time.time() - start_time

                # Track cost
                self._track_cost(result.cost, "scan")

                if result.status == "COMPLETED":
                    return result

                # Retry on failure
                if attempts < max_attempts:
                    time.sleep(RETRY_DELAY * attempts)
                    continue

                return result

            except requests.exceptions.Timeout:
                error = f"Timeout after {timeout}s"
                if attempts < max_attempts:
                    time.sleep(RETRY_DELAY * attempts)
                    continue
                return TinyFishResult(
                    url=url,
                    goal=goal,
                    status="TIMEOUT",
                    error=error,
                    duration=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )

            except requests.exceptions.RequestException as e:
                error = f"Request failed: {str(e)}"
                if attempts < max_attempts:
                    time.sleep(RETRY_DELAY * attempts)
                    continue
                return TinyFishResult(
                    url=url,
                    goal=goal,
                    status="FAILED",
                    error=error,
                    duration=time.time() - start_time,
                    timestamp=datetime.now().isoformat()
                )

        # Should never reach here, but just in case
        return TinyFishResult(
            url=url,
            goal=goal,
            status="FAILED",
            error=f"Max retries ({max_attempts}) exceeded",
            duration=time.time() - start_time,
            timestamp=datetime.now().isoformat()
        )

    def scan_batch(
        self,
        urls: List[str],
        goal: str,
        max_concurrent: int = 4,
        stealth: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[TinyFishResult]:
        """Scan multiple URLs in parallel.

        Args:
            urls: List of URLs to scan
            goal: Natural language description (same for all URLs)
            max_concurrent: Maximum concurrent scans (respects TinyFish limits)
            stealth: Use stealth mode
            progress_callback: Function called with (completed, total) after each scan

        Returns:
            List of TinyFishResults

        Example:
            def progress(completed, total):
                print(f"Progress: {completed}/{total}")

            results = client.scan_batch(
                bookmark_urls[:50],
                "Extract title, description, and category",
                max_concurrent=4,
                progress_callback=progress
            )
        """
        results = []
        completed = 0
        total = len(urls)

        def scan_with_progress(url):
            nonlocal completed
            result = self.scan_url(url, goal, stealth=stealth)
            completed += 1
            if progress_callback:
                progress_callback(completed, total)
            return result

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_url = {
                executor.submit(scan_with_progress, url): url
                for url in urls
            }

            for future in concurrent.futures.as_completed(future_to_url):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    url = future_to_url[future]
                    results.append(TinyFishResult(
                        url=url,
                        goal=goal,
                        status="FAILED",
                        error=f"Execution error: {str(e)}",
                        timestamp=datetime.now().isoformat()
                    ))

        return results

    def extract_competitive_intel(self, url: str) -> TinyFishResult:
        """Extract competitive intelligence from a competitor's website.

        Specialized extraction for:
          - Pricing tiers and limits
          - Feature lists
          - Changelog/release notes
          - Tech stack indicators

        Args:
            url: Competitor website URL

        Returns:
            TinyFishResult with structured competitive data
        """
        goal = """
        Extract competitive intelligence:
        1. Pricing (tiers, prices, limits, features per tier)
        2. Key features and differentiators
        3. Tech stack indicators (React, Vue, APIs mentioned)
        4. Latest updates/changelog (if visible)
        5. Team size indicators
        6. Customer testimonials or case studies

        Return as JSON with keys: pricing, features, tech_stack, updates, team_size, testimonials
        """
        return self.scan_url(url, goal, stealth=True, timeout=180)

    def monitor_github_repo(
        self,
        repo_url: str,
        events: Optional[List[str]] = None
    ) -> TinyFishResult:
        """Monitor a GitHub repository for specific events.

        Args:
            repo_url: GitHub repo URL (e.g., https://github.com/openclaw/openclaw)
            events: Events to extract (default: ["commits", "releases", "prs"])

        Returns:
            TinyFishResult with latest repo activity
        """
        events = events or ["commits", "releases", "prs"]

        # Navigate to latest release page for most repos
        if "/releases" not in repo_url:
            scan_url = f"{repo_url.rstrip('/')}/releases/latest"
        else:
            scan_url = repo_url

        goal = f"""
        Extract latest activity from GitHub repo:
        - Latest release version and date
        - Release notes and breaking changes
        - Recent commits (last 5)
        - Open PRs with descriptions
        - Stars and forks count

        Return as JSON with keys: latest_release, recent_commits, open_prs, stars, forks
        """

        return self.scan_url(scan_url, goal, stealth=False, timeout=90)

    def _parse_sse_stream(
        self,
        response: requests.Response,
        url: str,
        goal: str
    ) -> TinyFishResult:
        """Parse Server-Sent Events stream from TinyFish API.

        Args:
            response: Streaming HTTP response
            url: Original URL
            goal: Original goal

        Returns:
            TinyFishResult with parsed data
        """
        steps = 0
        last_event = None

        try:
            for line in response.iter_lines():
                if not line:
                    continue

                line = line.decode('utf-8')

                # SSE format: "data: {json}"
                if line.startswith('data: '):
                    data_str = line[6:]  # Remove "data: " prefix
                    try:
                        event = json.loads(data_str)
                        last_event = event

                        if event.get('type') == 'STEP':
                            steps += 1

                        elif event.get('type') == 'COMPLETE':
                            status = event.get('status', 'UNKNOWN')
                            result_json = event.get('resultJson')

                            return TinyFishResult(
                                url=url,
                                goal=goal,
                                status=status,
                                result_json=result_json,
                                steps=steps,
                                cost=steps * COST_PER_STEP,
                                timestamp=datetime.now().isoformat()
                            )

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            return TinyFishResult(
                url=url,
                goal=goal,
                status="FAILED",
                error=f"Stream parsing error: {str(e)}",
                steps=steps,
                cost=steps * COST_PER_STEP,
                timestamp=datetime.now().isoformat()
            )

        # If we get here, stream ended without COMPLETE event
        return TinyFishResult(
            url=url,
            goal=goal,
            status="FAILED",
            error="Stream ended without completion event",
            steps=steps,
            cost=steps * COST_PER_STEP,
            timestamp=datetime.now().isoformat()
        )

    def _track_cost(self, cost: float, operation: str):
        """Track API costs in WeEvolve cost log.

        Args:
            cost: Cost in USD
            operation: Operation type ("scan", "batch", "competitive")
        """
        try:
            # Load existing cost log
            if COST_LOG.exists():
                tracker = json.loads(COST_LOG.read_text())
            else:
                tracker = {
                    'daily_totals': {},
                    'tinyfish_total': 0.0,
                    'tinyfish_scans': 0
                }

            # Update costs
            today = datetime.now().strftime('%Y-%m-%d')
            daily = tracker.get('daily_totals', {})
            daily[today] = daily.get(today, 0.0) + cost
            tracker['daily_totals'] = daily
            tracker['tinyfish_total'] = tracker.get('tinyfish_total', 0.0) + cost
            tracker['tinyfish_scans'] = tracker.get('tinyfish_scans', 0) + 1

            # Save
            COST_LOG.parent.mkdir(parents=True, exist_ok=True)
            COST_LOG.write_text(json.dumps(tracker, indent=2))

        except Exception as e:
            # Don't fail on cost tracking errors
            print(f"Warning: Cost tracking failed: {e}")


# ============================================================================
# SCAN FOR USER — Proxied TinyFish via our API key
# ============================================================================

# Rate limits for user-facing scans (enforced locally)
FREE_SCANS_PER_DAY = 5
PRO_SCANS_PER_DAY = -1  # unlimited

TINYFISH_RESULTS_DIR = DATA_DIR / "tinyfish_results"

# SEED extraction prompt for TinyFish results
_SEED_GOAL = """
Extract structured knowledge from this page. Return JSON with these fields:
- title: Brief title of the content
- perceive: What are the key facts?
- connect: How does this connect to AI agents, development tools, or growth?
- learn: What is the ONE key actionable takeaway?
- question: What assumption should be challenged?
- expand: What opportunity does this reveal?
- share: What insight should be shared?
- receive: What feedback does this give about our approach?
- improve: How should this change how we operate?
- skills: Array of relevant skill tags
- quality: Float 0.0-1.0 (how actionable is this?)
- is_alpha: Boolean (is this a competitive advantage?)
- alpha_type: String or null (e.g. "tool", "strategy", "insight")
- key_entities: Array of key entities mentioned
- connections: Array of how this connects to existing knowledge
"""


def _get_user_scan_count_today() -> int:
    """Get how many TinyFish scans the user has done today."""
    usage_path = DATA_DIR / "tinyfish_usage.json"
    today = datetime.now().strftime("%Y-%m-%d")

    if not usage_path.exists():
        return 0

    try:
        with open(usage_path) as f:
            usage = json.load(f)
        return usage.get("daily", {}).get(today, 0)
    except (json.JSONDecodeError, OSError):
        return 0


def _record_user_scan():
    """Record that the user performed a TinyFish scan."""
    usage_path = DATA_DIR / "tinyfish_usage.json"
    today = datetime.now().strftime("%Y-%m-%d")

    usage = {}
    if usage_path.exists():
        try:
            with open(usage_path) as f:
                usage = json.load(f)
        except (json.JSONDecodeError, OSError):
            usage = {}

    daily = usage.get("daily", {})
    daily[today] = daily.get(today, 0) + 1
    total = usage.get("total_scans", 0) + 1

    updated_usage = {
        **usage,
        "daily": daily,
        "total_scans": total,
        "last_scan": datetime.now().isoformat(),
    }

    usage_path.parent.mkdir(parents=True, exist_ok=True)
    with open(usage_path, "w") as f:
        json.dump(updated_usage, f, indent=2)


def _is_pro_user() -> bool:
    """Check if the user has a Pro subscription."""
    try:
        from weevolve.tiers import get_current_tier
        tier = get_current_tier()
        return tier == "pro"
    except ImportError:
        return False


def _save_result_for_genesis(result: 'TinyFishResult', seed_data: dict):
    """Save a TinyFish result so it feeds back into the genesis pipeline.

    This is the collective loop: user scan -> genesis -> all users benefit.
    """
    TINYFISH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    result_id = result.url.replace("https://", "").replace("http://", "")
    result_id = re.sub(r'[^a-zA-Z0-9_-]', '_', result_id)[:80]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{result_id}.json"

    entry = {
        **seed_data,
        "id": f"tinyfish_{timestamp}_{result_id}",
        "source_url": result.url,
        "source_type": "tinyfish",
        "scan_cost": result.cost,
        "scan_steps": result.steps,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    filepath = TINYFISH_RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(entry, f, indent=2)

    return filepath


def scan_for_user(
    url: str,
    goal: Optional[str] = None,
) -> Dict:
    """
    Scan a URL using OUR TinyFish API key, on behalf of the user.

    This lets users benefit from TinyFish without their own key.
    Results go into:
      1. The user's local WeEvolve DB (immediate benefit)
      2. The genesis pipeline (collective benefit for all users)

    Rate limits:
      - Free: 5 scans/day
      - Pro: unlimited

    Args:
        url: URL to scan
        goal: Custom extraction goal (defaults to SEED knowledge extraction)

    Returns:
        Dict with scan result + integration status
    """
    # Check rate limit
    is_pro = _is_pro_user()
    scans_today = _get_user_scan_count_today()
    daily_limit = PRO_SCANS_PER_DAY if is_pro else FREE_SCANS_PER_DAY

    if daily_limit > 0 and scans_today >= daily_limit:
        remaining = 0
        tier_name = "Pro" if is_pro else "Free"
        return {
            "status": "RATE_LIMITED",
            "error": f"{tier_name} tier limit reached ({daily_limit}/day). "
                     f"{'Upgrade to Pro for unlimited.' if not is_pro else 'Try again tomorrow.'}",
            "scans_today": scans_today,
            "daily_limit": daily_limit,
            "remaining": remaining,
        }

    # Use SEED extraction goal by default
    scan_goal = goal or _SEED_GOAL

    # Create client using our API key (from env)
    try:
        client = TinyFishClient()
    except ValueError:
        return {
            "status": "NO_API_KEY",
            "error": "TinyFish API key not configured on server.",
        }

    # Perform the scan
    result = client.scan_url(url, scan_goal, stealth=True)

    # Record usage
    _record_user_scan()
    scans_today += 1
    remaining = max(0, daily_limit - scans_today) if daily_limit > 0 else -1

    if result.status != "COMPLETED" or not result.result_json:
        return {
            "status": result.status,
            "error": result.error or "Scan did not complete",
            "url": url,
            "cost": result.cost,
            "duration": result.duration,
            "scans_today": scans_today,
            "remaining": remaining,
        }

    # Parse the result as SEED knowledge
    seed_data = result.result_json
    if isinstance(seed_data, str):
        try:
            seed_data = json.loads(seed_data)
        except (json.JSONDecodeError, TypeError):
            seed_data = {"learn": seed_data, "quality": 0.5}

    # Ensure required fields
    seed_data.setdefault("title", f"TinyFish scan: {url[:60]}")
    seed_data.setdefault("quality", 0.7)
    seed_data.setdefault("skills", [])
    seed_data.setdefault("is_alpha", False)

    # Save to user's local DB via the learn pipeline
    integrated = False
    try:
        from weevolve.core import init_db, store_knowledge_atom
        db = init_db()
        raw_content = json.dumps(seed_data)
        atom_id = store_knowledge_atom(db, seed_data, raw_content, url, "tinyfish")
        integrated = atom_id is not None
    except Exception:
        # Fallback: save raw result for later processing
        integrated = False

    # Save for genesis pipeline (collective benefit)
    genesis_path = None
    try:
        genesis_path = str(_save_result_for_genesis(result, seed_data))
    except Exception:
        pass

    return {
        "status": "COMPLETED",
        "url": url,
        "title": seed_data.get("title", ""),
        "learn": seed_data.get("learn", ""),
        "quality": seed_data.get("quality", 0),
        "is_alpha": seed_data.get("is_alpha", False),
        "skills": seed_data.get("skills", []),
        "integrated_to_local_db": integrated,
        "saved_for_genesis": genesis_path is not None,
        "cost": result.cost,
        "duration": result.duration,
        "scans_today": scans_today,
        "remaining": remaining,
    }


# Need re and timezone for _save_result_for_genesis
import re
from datetime import timezone


# ============================================================================
# CLI Interface
# ============================================================================

def main():
    """CLI for testing TinyFish integration."""
    import sys

    if len(sys.argv) < 3:
        print(__doc__)
        print("\nUsage:")
        print("  python3 -m weevolve.tinyfish scan <url> <goal>")
        print("  python3 -m weevolve.tinyfish competitive <url>")
        print("  python3 -m weevolve.tinyfish github <repo_url>")
        print("\nExamples:")
        print('  python3 -m weevolve.tinyfish scan "https://example.com" "Extract pricing"')
        print('  python3 -m weevolve.tinyfish competitive "https://openclaw.com"')
        print('  python3 -m weevolve.tinyfish github "https://github.com/openclaw/openclaw"')
        sys.exit(1)

    command = sys.argv[1]

    try:
        client = TinyFishClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nSet TINYFISH_API_KEY environment variable:")
        print("  export TINYFISH_API_KEY=your_api_key_here")
        sys.exit(1)

    if command == "scan":
        url = sys.argv[2]
        goal = sys.argv[3] if len(sys.argv) > 3 else "Extract all text content"

        print(f"Scanning: {url}")
        print(f"Goal: {goal}")
        print("Streaming progress...\n")

        result = client.scan_url(url, goal)

        print(f"\nStatus: {result.status}")
        print(f"Steps: {result.steps}")
        print(f"Cost: ${result.cost:.4f}")
        print(f"Duration: {result.duration:.1f}s")

        if result.status == "COMPLETED" and result.result_json:
            print("\nResult:")
            print(json.dumps(result.result_json, indent=2))
        elif result.error:
            print(f"\nError: {result.error}")

    elif command == "competitive":
        url = sys.argv[2]

        print(f"Extracting competitive intel from: {url}")
        print("This may take 2-3 minutes...\n")

        result = client.extract_competitive_intel(url)

        print(f"\nStatus: {result.status}")
        print(f"Steps: {result.steps}")
        print(f"Cost: ${result.cost:.4f}")
        print(f"Duration: {result.duration:.1f}s")

        if result.status == "COMPLETED" and result.result_json:
            print("\nCompetitive Intelligence:")
            print(json.dumps(result.result_json, indent=2))
        elif result.error:
            print(f"\nError: {result.error}")

    elif command == "github":
        repo_url = sys.argv[2]

        print(f"Monitoring GitHub repo: {repo_url}")
        print("Fetching latest activity...\n")

        result = client.monitor_github_repo(repo_url)

        print(f"\nStatus: {result.status}")
        print(f"Steps: {result.steps}")
        print(f"Cost: ${result.cost:.4f}")
        print(f"Duration: {result.duration:.1f}s")

        if result.status == "COMPLETED" and result.result_json:
            print("\nRepo Activity:")
            print(json.dumps(result.result_json, indent=2))
        elif result.error:
            print(f"\nError: {result.error}")

    else:
        print(f"Unknown command: {command}")
        print("Valid commands: scan, competitive, github")
        sys.exit(1)


if __name__ == "__main__":
    main()
