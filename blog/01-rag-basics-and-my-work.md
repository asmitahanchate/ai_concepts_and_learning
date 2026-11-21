# What I learned about RAG, and how it connects to my work

I learned something new today, and I want to write it down while it's fresh.

At DevRev, I build AI agents for our customers. Twice this year, an agent I worked on gave answers that sounded confident but weren't true. In both cases the model itself was fine. The problem was simpler: **the model never saw the right information.**

Today I took a step back, went over the basics properly, and then built a small project to test what I learned. Here's what I found.

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

## Where I had already seen this at work

### 1. BILL: a developer chatbot that invented API fields

BILL's developer chatbot was answering API questions, but sometimes it **made up fields** that didn't exist, or gave endpoint examples that didn't match the real API.

- BILL's API specification is about **800,000 characters**, far too big to give the model all at once.
- The plan was to split the spec **one piece per API operation**. This is chunking by the document's natural structure.
- The agent used two steps:
  - **Discover:** find the API operation that matches the developer's question.
  - **Verify:** look up that operation's **exact** definition (parameters, request body, responses) before answering.
- The rule: no technical details without a successful lookup, and if the docs and the spec disagree, **the spec wins**.
- Later, I fixed wrong documentation links by building each link from the endpoint details.

**Something that didn't work, and what I learned from it:** I first tried asking an LLM to enrich each endpoint while building the index (about 300 endpoints, with 10 workers running at once). It hit rate limits, with waits of 39 to 293 seconds, and the job timed out at its 120-second limit. Lesson: **be careful about calling an LLM for every item while indexing.** Use a queue, retries with backoff, and a time budget.

### 2. Nutanix: build logs too big to search

Nutanix's root-cause-analysis agent needed to read build logs attached to support tickets. Some logs were up to **100 MB**, and files over **1 MB** were not indexed automatically, so the agent couldn't search them at all.

- I built a chunker that downloads the log from the ticket and splits it into **~60 KB pieces**, saving each piece as a searchable article with tags.
- A 100 MB log became about **1,700 articles**.
- Creating that many articles meant many API calls, so the bot ran into **rate limits**. A change was proposed to raise its limit from 2,000 to 10,000 requests per minute.
- The RCA agent then answered from these chunked articles.

**What I understood today:** the platform doesn't turn a 60 KB article into one embedding. It splits each article again into smaller passages before embedding them, and shows the full article when one of its passages matches. So without planning it, I had built **two-level chunking**: my code made the big pieces, and the platform made the small ones.

### 3. Camping World: when search isn't the right tool

For a Camping World agent, a user wanted to find a specific chat. Search returned the **top 10 out of 223** chat rooms, but the one they needed was ranked **#143**, so it never showed up. I switched that part to a database query (SQL). Lesson: **search gives you the best matches, not a complete list.** For "find all of X," use a query.

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

**1. Smaller pieces cost much less.** Cutting the pages sent **9 to 18 times less text** and still found 63% to 80% of the answers. For BILL, the full spec was too big to use. For Nutanix, the logs couldn't even be indexed. Cutting things down wasn't optional.

**2. Cutting along the document's own structure worked best.** Splitting at section headings, with the heading on top of each piece, won with both kinds of search. Each piece is one complete idea with a name. That's the same idea as splitting BILL's spec one operation at a time. My Nutanix chunker split by size, so next time I would split logs at build stages or error blocks, with a label like "build 412 > test stage > error."

**3. Meaning search usually beat keyword search.** People ask questions in different words than the documents use, and meaning search handles that well. Keyword search still matters for exact values like build numbers, error codes, or field names like `billCurrency`.

**4. Neither search was perfect, and they missed different questions.** That's why combining them (hybrid search) works so well in practice, and it's what I'll test next.

**5. More pieces means more cost.** The "small pieces" approach made 4,584 pieces, about four times more to embed and store. I saw the same thing with Nutanix: more chunks meant more API calls and more rate limits.

**6. Running a model during indexing is slow.** Cutting by topic change was the slowest method, because it runs the embedding model on every sentence. It's the same lesson as my BILL indexing attempt.

**7. Testing is the most important part.** Without the 30 questions, I would only be guessing which method was better. For BILL, I asked for a bug bash and a set of example questions with expected answers. For Nutanix, I reported about 96% accuracy, and now I know how I would measure a number like that more carefully: known failures, the error line I expect to find, and a score for each.

---

## What I would do differently now

1. **Split by structure, not just size:** logs by stage, API specs by operation, docs by heading. Give every piece a label.
2. **Use keyword and meaning search together.**
3. **Measure before and after any change**, even with a small set of questions.
4. **Keep LLM calls out of indexing**, unless they're queued, retried, and time-limited.

---

## What's next

Next, I'll compare embedding models, try a few vector databases, and then combine keyword and meaning search with reranking. I'll write about each step as I go.

Thank you for reading. If you've worked on any of this, I'd love to hear what has worked for you.
