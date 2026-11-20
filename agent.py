import asyncio
import json
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List

import gspread
import requests
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from google import genai
from google.genai import types
from google.oauth2.service_account import Credentials
from pydantic import BaseModel, Field

# ==========================================
# 1. CUSTOMIZE YOUR PREFERENCES HERE
# ==========================================
SHEET_NAME = "Job Hunt Engine"
SENDER_EMAIL = os.environ["USER_EMAIL"]
RECEIVER_EMAIL = os.environ["USER_EMAIL"]

USER_PREFERENCES = """
- Target Domains & Roles: Data Analyst, Business Analyst, Data Engineer, Analytics Engineer, BI Developer, SQL Developer, Product Analyst, Data Scientist.
- Core Skills to Look For: SQL, Python, Data Warehousing, ETL, Tableau, Power BI, Looker, Snowflake, BigQuery, Spark, Data Modeling.
- Experience Level: Mid-level to Senior (Exclude internships, unpaid roles, or VP/Director executive roles).
- Preferred Locations (ANY of these match): Bengaluru, Gurugram, Delhi NCR, Hyderabad, Pune, Mumbai, Remote, Hybrid (India).
"""

# ==========================================
# 2. STRUCTURED OUTPUT SCHEMA FOR GEMINI
# ==========================================
class JobOpening(BaseModel):
    job_title: str = Field(description="Exact title of the job opening")
    location: str = Field(description="City, country, or Remote/Hybrid status listed")
    direct_apply_url: str = Field(description="Full URL link to the specific job posting. If relative, combine with base domain.")
    match_score: int = Field(description="Relevance score from 1 to 10 based on Data/SQL role fit and location match")
    why_it_matches: str = Field(description="1-sentence explanation of why this fits the Data/SQL and location criteria")

class CareerPageExtraction(BaseModel):
    matching_jobs: List[JobOpening] = Field(default_factory=list)

# ==========================================
# 3. GOOGLE SHEETS CONNECTION
# ==========================================
def get_sheets():
    creds_dict = json.loads(os.environ["GCP_CREDENTIALS_JSON"])
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    client = gspread.authorize(creds)
    workbook = client.open(SHEET_NAME)
    return workbook.worksheet("Targets"), workbook.worksheet("Jobs_Found")

# ==========================================
# 4. WEB SCRAPER (CRAWL4AI + JINA FALLBACK)
# ==========================================
async def scrape_career_page(crawler: AsyncWebCrawler, url: str) -> str:
    run_config = CrawlerRunConfig(
        wait_until="networkidle",
        page_timeout=30000,
        remove_overlay_elements=True,
    )
    try:
        result = await crawler.arun(url=url, config=run_config)
        markdown_text = result.markdown or ""
    except Exception as e:
        print(f"Crawl4AI error on {url}: {e}")
        markdown_text = ""

    if len(markdown_text.strip()) < 250:
        print(f"Switching to Jina Reader fallback for {url}...")
        try:
            resp = requests.get(f"https://r.jina.ai/{url}", timeout=25)
            if resp.status_code == 200:
                markdown_text = resp.text
        except Exception as e:
            print(f"Fallback failed for {url}: {e}")

    return markdown_text[:80000]

# ==========================================
# 5. AI AGENT FILTERING (GEMINI FLASH)
# ==========================================
def extract_matching_jobs(ai_client: genai.Client, company: str, base_url: str, page_markdown: str) -> List[JobOpening]:
    if not page_markdown.strip():
        return []

    prompt = f"""
    You are an expert technical recruiter AI agent. Analyze the scraped career page content for **{company}** (Base URL: {base_url}).
    
    Identify ALL active job postings that match the user's preferences below:
    {USER_PREFERENCES}
    
    Strict Rules:
    1. Only include jobs that genuinely match BOTH the Data/SQL role criteria AND one of the preferred locations (or Remote).
    2. Convert relative job links (like '/careers/job/123') into full valid URLs using the Base URL '{base_url}'. If no specific job link exists, use '{base_url}'.
    3. Only return jobs with a match_score of 7 or higher.
    
    Scraped Career Page Content:
    {page_markdown}
    """

    try:
        response = ai_client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=CareerPageExtraction,
                temperature=0.1,
            ),
        )
        parsed = CareerPageExtraction.model_validate_json(response.text)
        return parsed.matching_jobs
    except Exception as e:
        print(f"Gemini extraction error for {company}: {e}")
        return []

# ==========================================
# 6. EMAIL NOTIFICATION (GMAIL SMTP)
# ==========================================
def send_email_alert(new_jobs: list):
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚀 {len(new_jobs)} New Data/SQL Job(s) Found Today!"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    rows_html = ""
    for job in new_jobs:
        rows_html += f"""
        <tr>
            <td style="padding: 10px; border: 1px solid #ddd;"><b>{job['company']}</b></td>
            <td style="padding: 10px; border: 1px solid #ddd;"><a href="{job['url']}">{job['title']}</a></td>
            <td style="padding: 10px; border: 1px solid #ddd;">{job['location']}</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{job['score']}/10</td>
            <td style="padding: 10px; border: 1px solid #ddd;">{job['reason']}</td>
        </tr>
        """

    html_body = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color: #333;">
        <h2>New Relevant Jobs Matched Your Criteria</h2>
        <p>Your AI Job Agent scanned your target career portals and found the following new openings:</p>
        <table style="border-collapse: collapse; width: 100%; text-align: left;">
          <thead>
            <tr style="background-color: #f4f6f8;">
              <th style="padding: 10px; border: 1px solid #ddd;">Company</th>
              <th style="padding: 10px; border: 1px solid #ddd;">Role</th>
              <th style="padding: 10px; border: 1px solid #ddd;">Location</th>
              <th style="padding: 10px; border: 1px solid #ddd;">Score</th>
              <th style="padding: 10px; border: 1px solid #ddd;">AI Match Note</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
      </body>
    </html>
    """

    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(SENDER_EMAIL, os.environ["GMAIL_APP_PASSWORD"])
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
    print("Email notification sent successfully!")

# ==========================================
# 7. MAIN PIPELINE ORCHESTRATOR
# ==========================================
async def main():
    targets_sheet, found_sheet = get_sheets()
    targets = targets_sheet.get_all_records()
    existing_records = found_sheet.get_all_records()

    seen_signatures = {
        f"{str(row.get('Company', '')).lower()}|{str(row.get('Job Title', '')).lower()}"
        for row in existing_records
    }
    seen_urls = {str(row.get("Job Link", "")).strip() for row in existing_records}

    ai_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    browser_cfg = BrowserConfig(headless=True, enable_stealth=True)
    
    newly_discovered = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        for row in targets:
            company = str(row.get("Company Name", "")).strip()
            career_url = str(row.get("Career URL", "")).strip()
            if not company or not career_url:
                continue

            print(f"Scanning {company} -> {career_url}")
            markdown = await scrape_career_page(crawler, career_url)
            matches = extract_matching_jobs(ai_client, company, career_url, markdown)

            for job in matches:
                sig = f"{company.lower()}|{job.job_title.lower()}"
                is_duplicate = (sig in seen_signatures) or (
                    job.direct_apply_url != career_url and job.direct_apply_url in seen_urls
                )

                if not is_duplicate:
                    seen_signatures.add(sig)
                    seen_urls.add(job.direct_apply_url)

                    row_data = [
                        today_str,
                        company,
                        job.job_title,
                        job.location,
                        job.match_score,
                        job.direct_apply_url,
                        job.why_it_matches,
                    ]
                    found_sheet.append_row(row_data)

                    newly_discovered.append({
                        "company": company,
                        "title": job.job_title,
                        "location": job.location,
                        "score": job.match_score,
                        "url": job.direct_apply_url,
                        "reason": job.why_it_matches,
                    })
                    print(f"  [+] New Match: {job.job_title} ({job.location})")

    if newly_discovered:
        send_email_alert(newly_discovered)
    else:
        print("No new matching jobs found today.")

if __name__ == "__main__":
    asyncio.run(main())