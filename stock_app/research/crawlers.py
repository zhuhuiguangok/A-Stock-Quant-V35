"""研报爬虫（同步版）。

移植自原 FastAPI 项目的 async 爬虫（iyanbao/hibor/eastmoney/sina/fxbaogao/it199），
httpx 异步客户端改为同步 Client + ThreadPoolExecutor（在 services 层并发编排）。
接口约定：fetch_list() -> list[dict]，fetch_detail(url) -> dict。
"""
from __future__ import annotations

import json
import os
import random
import re
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import httpx
from bs4 import BeautifulSoup


MAX_CONTENT_LENGTH = 50000
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Mobile Safari/537.36",
]


def clean_text(text: str) -> str:
    return " ".join(text.split()) if text else ""


def _clear_proxy_env() -> None:
    import os

    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"


def download_and_parse_pdf(url: str, headers: dict) -> str:
    """在子进程中下载并解析 PDF。

    PyMuPDF 在部分 Windows 环境解析特定 PDF 会触发原生 access violation
    直接杀死整个进程，因此必须隔离在子进程里跑，崩溃/超时只损失该次解析。
    """
    script = (
        "import sys, re\n"
        "import httpx, fitz\n"
        "url, ua = sys.argv[1], sys.argv[2]\n"
        "with httpx.Client(timeout=60.0, follow_redirects=True, trust_env=False) as c:\n"
        "    r = c.get(url, headers={'User-Agent': ua})\n"
        "    r.raise_for_status()\n"
        "doc = fitz.open(stream=r.content, filetype='pdf')\n"
        "text = ''.join(p.get_text() for p in doc)\n"
        "doc.close()\n"
        "sys.stdout.write(re.sub(r'\\s+', ' ', text).strip()[:50000])\n"
    )
    import subprocess
    import sys

    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
        env.pop(key, None)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", script, url, headers.get("User-Agent", "")],
            capture_output=True, timeout=45, env=env, cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        if proc.returncode != 0:
            return ""
        return proc.stdout.decode("utf-8", errors="ignore")[:MAX_CONTENT_LENGTH]
    except Exception:
        return ""


class BaseCrawler:
    source_code: str = ""
    base_url: str = ""

    def build_headers(self) -> dict:
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Referer": "https://www.baidu.com/",
        }

    def _client(self, timeout: float = 30.0) -> httpx.Client:
        return httpx.Client(timeout=timeout, follow_redirects=True, trust_env=False)

    def fetch_list(self, page: int = 1) -> List[Dict]:
        raise NotImplementedError

    def fetch_detail(self, url: str) -> Dict:
        raise NotImplementedError


class IyanbaoCrawler(BaseCrawler):
    source_code = "iyanbao"
    base_url = "https://www.iyanbao.com"

    def fetch_list(self, page: int = 1) -> List[Dict]:
        all_reports, seen_ids = [], set()
        for p in (1, 2, 3, 4, 5):
            for url in (
                f"{self.base_url}/trade?page={p}",
                f"{self.base_url}/company?page={p}",
                f"{self.base_url}/strategy?page={p}",
            ):
                try:
                    with self._client() as client:
                        response = client.get(url, headers=self.build_headers())
                    for r in self._parse_list(response.text):
                        if r["external_id"] not in seen_ids:
                            seen_ids.add(r["external_id"])
                            all_reports.append(r)
                except Exception:
                    continue
        return all_reports[:150]

    def _parse_list(self, html: str) -> List[Dict]:
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', html, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(1))
        except Exception:
            return []
        search_list = data.get("props", {}).get("pageProps", {}).get("searchList", [])
        reports = []
        for item in search_list:
            try:
                inst = item.get("institution") or {}
                institution = inst.get("value", "") if isinstance(inst, dict) else str(inst)
                timestamp = item.get("publishTimestamp", 0)
                report_id = item.get("id", 0)
                if timestamp and timestamp > 10000000000:
                    d = datetime.fromtimestamp(timestamp / 1000)
                elif timestamp:
                    d = datetime.fromtimestamp(timestamp)
                else:
                    d = datetime.now()
                content = item.get("content", "")
                reports.append({
                    "external_id": f"iyanbao_{report_id}",
                    "title": item.get("title", ""),
                    "source_url": f"{self.base_url}/report/{report_id}",
                    "report_date": d.date(),
                    "institution": institution,
                    "content_text": (content or "")[:MAX_CONTENT_LENGTH],
                    "pdf_url": item.get("pdfUrl") or None,
                })
            except Exception:
                continue
        return reports

    def fetch_detail(self, url: str) -> Dict:
        with self._client() as client:
            response = client.get(url, headers=self.build_headers())
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text, re.DOTALL)
        content = ""
        if match:
            try:
                report_data = json.loads(match.group(1)).get("props", {}).get("pageProps", {}).get("reportData", {})
                if report_data:
                    content = report_data.get("content") or report_data.get("summary") or ""
            except Exception:
                pass
        return {"content_text": content[:MAX_CONTENT_LENGTH], "pdf_url": None}


class HiborCrawler(BaseCrawler):
    source_code = "hibor"
    base_url = "https://www.hibor.com.cn"

    def fetch_list(self, page: int = 1) -> List[Dict]:
        all_reports, seen = [], set()
        with self._client() as client:
            for p in range(1, 4):
                try:
                    response = client.get(f"{self.base_url}/freport_11_{p}.html", headers=self.build_headers())
                    for r in self._parse_list(response.text):
                        if r["external_id"] not in seen:
                            seen.add(r["external_id"])
                            all_reports.append(r)
                except Exception:
                    continue
        return all_reports

    def _parse_list(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "lxml")
        reports = []
        for a in soup.select("a[href^='/data/']"):
            try:
                title = clean_text(a.get_text(strip=True))
                if not title or len(title) < 10:
                    continue
                href = a.get("href", "")
                title = re.sub(r"^\d+[.、]\s*", "", title).strip()
                idx = title.find("证券")
                institution = title[: idx + 2] if idx > 0 else "慧博投研"
                full_url = f"{self.base_url}{href}"
                reports.append({
                    "external_id": full_url,
                    "title": title,
                    "source_url": full_url,
                    "report_date": date.today(),
                    "institution": institution,
                })
            except Exception:
                continue
        return reports

    def fetch_detail(self, url: str) -> Dict:
        with self._client() as client:
            response = client.get(url, headers=self.build_headers())
        soup = BeautifulSoup(response.text, "lxml")
        pdf_url = None
        for a in soup.find_all("a", href=True):
            if ".pdf" in a["href"]:
                href = a["href"]
                pdf_url = href if href.startswith("http") else f"{self.base_url}{href}"
                break
        if pdf_url:
            content = download_and_parse_pdf(pdf_url, self.build_headers())
            if content:
                return {"content_text": content, "pdf_url": pdf_url}
        return {"content_text": clean_text(soup.get_text())[:MAX_CONTENT_LENGTH], "pdf_url": pdf_url}


class EastmoneyCrawler(BaseCrawler):
    source_code = "eastmoney"
    base_url = "https://data.eastmoney.com"
    api_url = "https://reportapi.eastmoney.com/report/list"

    QTYPES = [0, 1, 2, 3, 4]
    PAGES_PER_TYPE = 3
    PAGE_SIZE = 50

    def fetch_list(self, page: int = 1) -> List[Dict]:
        begin = (date.today() - timedelta(days=7)).isoformat()
        end = date.today().isoformat()
        all_reports, seen = [], set()
        with self._client() as client:
            for qtype in self.QTYPES:
                for p in range(1, self.PAGES_PER_TYPE + 1):
                    try:
                        response = client.get(self.api_url, params={
                            "qType": str(qtype), "pageSize": str(self.PAGE_SIZE), "pageNo": str(p),
                            "fields": "", "industryCode": "*", "industry": "*", "rating": "*",
                            "ratingChange": "*", "beginTime": begin, "endTime": end,
                            "orgCode": "", "code": "*", "rcode": "",
                        }, headers=self.build_headers())
                        items = (json.loads(response.text).get("data")) or []
                        for item in items:
                            raw = self._to_raw(item, seen)
                            if raw:
                                all_reports.append(raw)
                        if len(items) < self.PAGE_SIZE:
                            break
                    except Exception:
                        continue
        return all_reports

    def _to_raw(self, item: dict, seen: set) -> Optional[Dict]:
        try:
            info_code = item.get("infoCode", "")
            if not info_code or info_code in seen:
                return None
            seen.add(info_code)
            title = (item.get("title") or "").strip()
            if not title:
                return None
            try:
                report_date = datetime.strptime((item.get("publishDate") or "")[:10], "%Y-%m-%d").date()
            except Exception:
                report_date = date.today()
            return {
                "external_id": f"eastmoney_{info_code}",
                "title": title,
                "source_url": f"{self.base_url}/report/info/{info_code}.html",
                "stock_code": item.get("stockCode") or None,
                "stock_name": item.get("stockName") or None,
                "institution": item.get("orgSName") or item.get("orgName") or None,
                "author": item.get("researcher") or None,
                "report_date": report_date,
                "pdf_url": f"https://pdf.dfcfw.com/pdf/H3_{info_code}_1.pdf",
            }
        except Exception:
            return None

    def fetch_detail(self, url: str) -> Dict:
        match = re.search(r"/info/([A-Za-z0-9]+)\.html", url)
        if match:
            pdf_url = f"https://pdf.dfcfw.com/pdf/H3_{match.group(1)}_1.pdf"
            content = download_and_parse_pdf(pdf_url, self.build_headers())
            if content:
                return {"content_text": content, "pdf_url": pdf_url}
        return {"content_text": "", "pdf_url": None}


class SinaCrawler(BaseCrawler):
    source_code = "sina"
    base_url = "http://stock.finance.sina.com.cn"

    def fetch_list(self, page: int = 1) -> List[Dict]:
        all_reports, seen = [], set()
        with self._client() as client:
            for p in range(1, 4):
                try:
                    response = client.get(
                        f"{self.base_url}/stock/go.php/vReport_List/kind/search/index.phtml?p={p}",
                        headers=self.build_headers(),
                    )
                    response.encoding = "gbk"
                    for r in self._parse_list(response.text):
                        if r["external_id"] not in seen:
                            seen.add(r["external_id"])
                            all_reports.append(r)
                except Exception:
                    continue
        return all_reports

    def _parse_list(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "lxml")
        reports = []
        for a in soup.select('a[href*="vReport_Show"]'):
            try:
                href = a.get("href", "")
                title = clean_text(a.get_text())
                if not href or not title or len(title) < 6:
                    continue
                id_match = re.search(r"rptid/(\d+)", href)
                if not id_match:
                    continue
                stock_code, stock_name = self._parse_stock(title)
                reports.append({
                    "external_id": f"sina_{id_match.group(1)}",
                    "title": title,
                    "source_url": href if href.startswith("http") else f"https:{href}",
                    "stock_code": stock_code,
                    "stock_name": stock_name,
                    "institution": self._parse_institution(title),
                    "report_date": self._find_date_near(a),
                })
            except Exception:
                continue
        return reports

    @staticmethod
    def _parse_stock(title: str):
        m = re.search(r"([一-龥A-Za-z]+)[(（](\d{6})[)）]", title)
        if m:
            return m.group(2), m.group(1)
        m2 = re.search(r"[(（](\d{6})[)）]", title)
        return (m2.group(1), None) if m2 else (None, None)

    @staticmethod
    def _parse_institution(title: str):
        m = re.search(r"^([^：:]{2,12}证券|[^：:]{2,12}基金|[^：:]{2,12}期货)[:：]", title)
        return m.group(1) if m else None

    @staticmethod
    def _find_date_near(a) -> date:
        parent = a
        for _ in range(4):
            parent = parent.parent
            if not parent:
                break
            m = re.search(r"(20\d{2}-\d{2}-\d{2})", parent.get_text())
            if m:
                try:
                    return datetime.strptime(m.group(1), "%Y-%m-%d").date()
                except Exception:
                    pass
        return date.today()

    def fetch_detail(self, url: str) -> Dict:
        with self._client() as client:
            response = client.get(url, headers=self.build_headers())
        response.encoding = "gbk"
        soup = BeautifulSoup(response.text, "lxml")
        content_div = soup.select_one("#content, .content, .report_content, .blk_container")
        content = clean_text(content_div.get_text()) if content_div else clean_text(soup.get_text())
        return {"content_text": content[:MAX_CONTENT_LENGTH], "pdf_url": None}


class FxbaogaoCrawler(BaseCrawler):
    source_code = "fxbaogao"
    base_url = "https://www.fxbaogao.com"

    def fetch_list(self, page: int = 1) -> List[Dict]:
        all_reports, seen = [], set()
        with self._client() as client:
            for url in (f"{self.base_url}/", f"{self.base_url}/report"):
                try:
                    response = client.get(url, headers=self.build_headers())
                    for r in self._parse_list(response.text):
                        if r["external_id"] not in seen:
                            seen.add(r["external_id"])
                            all_reports.append(r)
                except Exception:
                    continue
        return all_reports

    def _parse_list(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "lxml")
        reports = []
        for a in soup.select('a[href^="/detail/"]'):
            try:
                href = a.get("href", "")
                title = clean_text(a.get_text())
                if not href or not title or len(title) < 6:
                    continue
                id_match = re.search(r"/detail/(\d+)", href)
                if not id_match:
                    continue
                reports.append({
                    "external_id": f"fxbaogao_{id_match.group(1)}",
                    "title": title,
                    "source_url": f"{self.base_url}{href}",
                    "institution": "发现报告",
                    "report_date": date.today(),
                })
            except Exception:
                continue
        return reports

    def fetch_detail(self, url: str) -> Dict:
        with self._client() as client:
            response = client.get(url, headers=self.build_headers())
        content, pdf_url = "", None
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', response.text, re.DOTALL)
        if match:
            try:
                props = json.loads(match.group(1)).get("props", {}).get("pageProps", {})
                detail = props.get("data") or props.get("detail") or props.get("report") or {}
                if isinstance(detail, dict):
                    content = detail.get("abstract") or detail.get("summary") or detail.get("content") or detail.get("introduction") or ""
                    pdf_url = detail.get("pdfUrl") or detail.get("pdf_url") or detail.get("fileUrl")
            except Exception:
                pass
        if not content:
            content = clean_text(BeautifulSoup(response.text, "lxml").get_text())
        return {"content_text": content[:MAX_CONTENT_LENGTH], "pdf_url": pdf_url}


class It199Crawler(BaseCrawler):
    source_code = "it199"
    base_url = "https://www.199it.com"

    def fetch_list(self, page: int = 1) -> List[Dict]:
        all_reports, seen = [], set()
        with self._client() as client:
            for p in range(1, 3):
                try:
                    url = f"{self.base_url}/archives/category/report" if p == 1 else f"{self.base_url}/archives/category/report/page/{p}"
                    response = client.get(url, headers=self.build_headers())
                    for r in self._parse_list(response.text):
                        if r["external_id"] not in seen:
                            seen.add(r["external_id"])
                            all_reports.append(r)
                except Exception:
                    continue
        return all_reports

    def _parse_list(self, html: str) -> List[Dict]:
        soup = BeautifulSoup(html, "lxml")
        reports = []
        for a in soup.select('a[href*="/archives/"]'):
            try:
                href = a.get("href", "")
                title = clean_text(a.get_text())
                if not href or not title or len(title) < 8:
                    continue
                id_match = re.search(r"/archives/(\d+)\.html", href)
                if not id_match:
                    continue
                reports.append({
                    "external_id": f"it199_{id_match.group(1)}",
                    "title": title,
                    "source_url": href if href.startswith("http") else f"https:{href}",
                    "institution": "199IT",
                    "report_date": self._find_date_near(a),
                })
            except Exception:
                continue
        return reports

    @staticmethod
    def _find_date_near(a) -> date:
        parent = a
        for _ in range(4):
            parent = parent.parent
            if not parent:
                break
            m = re.search(r"(20\d{2}[年/-]\d{1,2}[月/-]\d{1,2})", parent.get_text())
            if m:
                try:
                    s = m.group(1).replace("年", "-").replace("月", "-").replace("/", "-").rstrip("日")
                    return datetime.strptime(s, "%Y-%m-%d").date()
                except Exception:
                    pass
        return date.today()

    def fetch_detail(self, url: str) -> Dict:
        with self._client() as client:
            response = client.get(url, headers=self.build_headers())
        soup = BeautifulSoup(response.text, "lxml")
        entry = soup.select_one(".entry-content, article")
        content = clean_text(entry.get_text()) if entry else clean_text(soup.get_text())
        pdf_link = soup.select_one("a[href$='.pdf']")
        return {"content_text": content[:MAX_CONTENT_LENGTH], "pdf_url": pdf_link.get("href") if pdf_link else None}


CRAWLER_REGISTRY = {
    cls.source_code: cls
    for cls in (IyanbaoCrawler, HiborCrawler, EastmoneyCrawler, SinaCrawler, FxbaogaoCrawler, It199Crawler)
}
