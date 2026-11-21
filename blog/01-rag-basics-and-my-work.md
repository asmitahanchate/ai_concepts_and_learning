# What I learned about RAG, and how it connects to my work

I learned something new today, and I want to write it down while it's fresh.

I'm an Applied AI Engineer at DevRev, working with S&P Global, Nutanix, Camping World and BILL. My work covers the whole path from data to answer: ingestion pipelines, retrieval and search, the agents on top, and the evaluation that tells us whether they work in production.

Again and again, I've seen the same lesson: when an AI system gives a wrong answer, the model is often fine. **The system never gave it the right information.** So today I went back over the fundamentals of retrieval properly, and built a small project to test them with real numbers. Here's what I found, and how it connects to my work.

---

## The basics, in simple words

Language models have three common problems:

1. **They can make things up.** This is called hallucination.
2. **They can only read so much at once.** This limit is called the context window.
3. **They don't know your private data.** They only know what they were trained on.

**RAG (Retrieval-Augmented Generation)** helps with all three. The idea is simple: find the right information first, put it in front of the model, then ask the question. The model stays the same. You're giving it an open book.

When the "book" is too big, you need a few more pieces:

- **Chunking:** cut the documents into smaller pieces.
- **Embeddings:** turn each piece into a list of numbers that represents its meaning. Pieces with similar meaning end up close together.
- **A vector database:** stores those numbers so you can quickly find the pieces closest to a question.
- **Keyword search:** meaning-based search can miss exact words like IDs, error codes or field names. Keyword search (BM25) finds them. Using both together is called **hybrid search**.

---

## Where these ideas show up in my work

### 1. BILL: a RAG pipeline over an 800,000-character API spec

BILL's developer assistant was sometimes **hallucinating API answers**: invented fields, or endpoint examples that didn't match the real API. Roughly 30–40% of answers had this problem.

- The OpenAPI specification is about **800,000 characters**, far too big to give the model all at once.
- I built a **Retrieval-Augmented Generation (RAG) pipeline** over the OpenAPI spec using **two-stage retrieval**:
  - **`/discover`** ranks API operations for a natural-language question.
  - **`/get_endpoint_details`** returns one operation's full schema (parameters, request body, responses).
- The assistant follows one simple rule: **verify before you answer.** No technical details without a successful lookup, and if the docs and the spec disagree, the spec wins.
- With **LLM-free indexing**, this cut hallucinations from **~30–40% to under 2%** and reduced context tokens by **~70%**, at sub-second latency.

**What I learned from a first attempt that didn't work:** I originally tried enriching all 300 endpoints with an LLM while indexing, using 10 workers. The calls hit rate-limit backoffs of up to 293 seconds, against a 120-second serverless runtime limit. So I switched to **LLM-free, operation-level indexing** instead, which was faster and fully reliable. Lesson: **be careful about putting an LLM call on every item in your indexing path.**

### 2. Nutanix: agent memory for build failures

At Nutanix, I architected **memory-based infrastructure for agents**, with Bring-Your-Own-Agent and Knowledge-Base support, so users can connect systems like Jira, Jenkins, AWS, GCP and Azure as agent memory.

- The **event-driven ingestion pipeline** captures each Jenkins build failure as it happens.
- Build logs can be up to **200 MB**, and files over **1 MB** weren't indexed automatically. So the pipeline chunks each log into **~60 KB segments**, which makes them searchable with hybrid **BM25 + vector search**. A 100 MB log becomes about 1,700 segments.
- On top of that, **signal-based text-to-SQL retrieval** finds the relevant failure data, and each failure is auto-routed to its owner as a Jira ticket.
- Result: build-failure triage went from **up to 7 hours to under 30 seconds**, with **~96% root-cause accuracy** across 15–30 failed builds a day.
- Creating that many segments also meant many API calls, so we had to plan carefully around **rate limits**.

**What I understood more deeply today:** the platform doesn't turn a 60 KB segment into one embedding. It splits each one again into smaller passages before embedding, and shows the full segment when a passage matches. So the system works as **two-level chunking**: big segments for context, small passages for precise matching.

### 3. Camping World: permission-scoped search, and knowing when not to search

For Camping World (~13,000 employees), I engineered an **enterprise search system** that unifies Outlook, Teams, Jira, Confluence and Asana behind one AI assistant, with **permission-scoped retrieval**.

- One user needed a specific Teams chat. Ranked search returned only the **top 10 of 223** chat rooms, and the one they needed was at **#143**.
- By routing exact lookups like this to **deterministic SQL**, result coverage went from **~4.5% to 100%**.
- Lesson: **search gives you the best matches, not a complete list.** For "find all of X," use a query.
- I also scaled Camping World's data agents with a **summary-table caching layer** over tens of millions of Snowflake records, replacing ~32-second warehouse scans that timed out agent workflows with pre-aggregated reads.

### 4. S&P Global: one knowledge layer for many sources

For S&P Global, I designed a **unified knowledge layer** that syncs data from any source (for example, Azure DevOps) so it can be searched and queried with SQL. This shrank quarterly program (PI) planning from **a week to a few hours** for release train engineers, scrum masters and PMs.

**The common thread:** in every case, the hard part wasn't the model. It was getting the **right data, in the right shape, to the right place**, reliably and with the right permissions.

---

## What I built today

I wanted to see these ideas with real numbers, so I built a small project:
**[github.com/asmitahanchate/ai_concepts_and_learning](https://github.com/asmitahanchate/ai_concepts_and_learning)**

- **Documents:** the public FastAPI documentation (114 pages).
- **Test:** 30 questions. Each one has a short "evidence" phrase that must appear in a search result for it to count as found.
- **Scores:**
  - **Recall@5:** was the answer in the top 5 results?
  - **MRR:** how high up was it?
  - **Context size:** how much text would be sent to the model?

### Project 00: Write the test first

Before building anything, I wrote the 30 test questions and a small checker to make sure every answer really exists in the documents.

Then I tried the simplest approach: keyword search over **whole pages**. It found every answer (Recall@5 = 1.00), but sent about **38,500 characters** to the model per question. That's like answering a question by handing someone five whole chapters.

### Project 01: Five ways to cut the documents

I cut the same pages five different ways and tested each one with both keyword search (BM25) and meaning search (the BGE-small embedding model):

| How I cut the documents | Pieces | Keyword search, Recall@5 | Meaning search, Recall@5 | Text sent |
|---|---:|---:|---:|---:|
| Every 500 characters | 1,301 | 0.67 | 0.77 | ~2,400 |
| At natural breaks (paragraphs, sentences) | 1,570 | 0.63 | 0.73 | ~2,000 |
| **At section headings, with the heading as a label** | 1,049 | **0.77** | **0.80** | ~3,600 |
| Where the topic changes | 1,084 | 0.73 | 0.77 | ~4,300 |
| Search small pieces, return the bigger section | 4,584 | 0.63 | 0.73 | ~4,400 |

---

## What the results taught me

**1. Smaller pieces cost much less.** Cutting the pages sent **9 to 18 times less text** and still found 63% to 80% of the answers. For BILL, the full spec was too big to use. For Nutanix, the raw logs couldn't even be indexed. Cutting things down wasn't optional.

**2. Cutting along the document's own structure worked best.** Splitting at section headings, with the heading on top of each piece, won with both kinds of search. Each piece is one complete idea with a name. That's the same idea as splitting BILL's spec one operation at a time. For logs, the next improvement I'd try is splitting at build stages or error blocks, with a label like "build 412 > test stage > error."

**3. Meaning search usually beat keyword search.** People ask questions in different words than the documents use, and meaning search handles that well. Keyword search still matters for exact values like build numbers, error codes, or field names like `billCurrency`.

**4. Neither search was perfect, and they missed different questions.** That's why combining them (hybrid search) works so well in practice, and it's what I'll test next.

**5. More pieces means more cost.** The "small pieces" approach made 4,584 pieces, about four times more to embed and store. I saw the same thing at Nutanix: more segments meant more API calls and more rate-limit planning.

**6. Running a model during indexing is slow.** Cutting by topic change was the slowest method, because it runs the embedding model on every sentence. It's the same lesson as BILL: LLM-free indexing was faster and more reliable than calling a model for every item.

**7. Testing is the most important part.** Without the 30 questions, I would only be guessing which method was better. This is why I co-built **EvalFrame**, an LLM evaluation framework that auto-generates test suites, replays multi-turn conversations against live agents, and scores them with a weighted LLM-as-judge rubric. This project reminded me that simple, mechanical checks ("is the answer actually in the result?") are a great partner to an LLM judge.

---

## What I'm taking away

1. **Split by structure, not just size:** logs by stage, API specs by operation, docs by heading. Give every piece a label.
2. **Use keyword and meaning search together.**
3. **Measure before and after any change**, even with a small set of questions.
4. **Keep LLM calls out of indexing**, unless they're queued, retried, and time-limited.

---

## What's next

Next, I'll compare embedding models, try a few vector databases, and then combine keyword and meaning search with reranking. I'll write about each step as I go.

Thank you for reading. If you've worked on any of this, I'd love to hear what has worked for you.
