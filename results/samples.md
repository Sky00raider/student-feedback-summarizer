# Sample outputs

## synthetic test (held-out phrasings + courses)

**Comment:** I felt that being allowed to pick our own project topic was motivating.

- Reference: `assignments | positive | Found choosing their own project topic motivating.`
- Keyword baseline (no model): `assignments | positive | I felt that being allowed to pick our own project topic was motivating.`
- distilgpt2, 3-shot prompt, no LoRA: `materials | mixed | Feels 18 hours a week is fair but says the expensive textbook was barely used.`
- distilgpt2 + LoRA, merged: `assignments | positive | Liked that being allowed to pick a project topic motivated by the project topic.`

**Comment:** so yeah there was no help available in the week before the exam

- Reference: `support | negative | Wants help available in the week before the exam.`
- Keyword baseline (no model): `assessment | negative | so yeah there was no help available in the week before the exam`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | the lecture notes were not very well written.`
- distilgpt2 + LoRA, merged: `support | negative | Says help was unavailable in the week before the exam.`

**Comment:** While the slower review weeks before exams really helped, the rubric for the essay was only shared after grading.

- Reference: `assessment | mixed | Valued the slower review weeks before exams but wants the essay rubric shared before grading.`
- Keyword baseline (no model): `assessment | positive | While the slower review weeks before exams really helped, the rubric for the essay was only shared after grading.`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | content | mixed | Found the forum active with fast answers but finds the hypothesis testing reading load unrealistic.`
- distilgpt2 + LoRA, merged: `assessment | mixed | Liked that the slower review weeks before exams helped, but says rubric shared after grading.`

**Comment:** the cheat sheet summarising each week was a lifesaver however there was no help available in the week before the exam

- Reference: `support | mixed | Values the weekly summary cheat sheet but wants help available in the week before the exam.`
- Keyword baseline (no model): `assessment | negative | the cheat sheet summarising each week was a lifesaver however there was no help available in the week before the`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | the lecture notes were not well written and the lecture notes were not well written.`
- distilgpt2 + LoRA, merged: `support | mixed | Says the cheat sheet summarising each week was a lifesaver but could not help at the time.`

**Comment:** the slower review weeks before exams really helped tbh

- Reference: `pacing | positive | Valued the slower review weeks before exams.`
- Keyword baseline (no model): `assessment | positive | the slower review weeks before exams really helped tbh`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | the book was not well-received.`
- distilgpt2 + LoRA, merged: `pacing | positive | Liked that slower reviews helped the slower review weeks before exams.`

**Comment:** deadlines from three deliverables landed in the same week

- Reference: `workload | negative | Wants deliverables spread out instead of clustered in one week.`
- Keyword baseline (no model): `workload | positive | deadlines from three deliverables landed in the same week`
- distilgpt2, 3-shot prompt, no LoRA: `materials | mixed | Feels 18 hours a week is fair but says the expensive textbook was barely used.`
- distilgpt2 + LoRA, merged: `workload | negative | Wants three deliverables delivered in the same week.`

**Comment:** the short recap at the start of each lecture helped a lot

- Reference: `lectures | positive | Values the recap at the start of each lecture.`
- Keyword baseline (no model): `lectures | positive | the short recap at the start of each lecture helped a lot`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | the lecture notes and the lecture notes were very good.`
- distilgpt2 + LoRA, merged: `lectures | positive | Liked the short recap at the start of each lecture.`

**Comment:** The cheat sheet summarising each week was a lifesaver. Great job.

- Reference: `materials | positive | Values the weekly summary cheat sheet.`
- Keyword baseline (no model): `materials | positive | The cheat sheet summarising each week was a lifesaver.`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | the lecture notes on the French Revolution lecture notes but finds grading inconsistent between TAs.`
- distilgpt2 + LoRA, merged: `assignments | positive | Praises the week cheat sheet summarising each week.`

## freeform (48 comments written independently)

**Comment:** The guest speakers were fantastic!! Loved them. The regular lectures by contrast are two hours of one-way talking with zero discussion, which is hard to sit through.

- Reference: `lectures | mixed | Loved the guest speakers but found regular two-hour lectures one-way with no discussion.`
- Keyword baseline (no model): `workload | negative | The guest speakers were fantastic!!`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | lecture | mixed | Found the forum active with fast answers but finds the hypothesis testing reading load unrealistic.`
- distilgpt2 + LoRA, merged: `lectures | mixed | Praises the regular lectures.`

**Comment:** Love the topic and the real datasets. The weekly problem sets alone are okay but with the quiz AND the discussion post AND the group memo all landing in the same week, it was too much.

- Reference: `workload | mixed | Enjoys the real datasets but struggled when quiz, discussion post and group memo all landed in the same week.`
- Keyword baseline (no model): `workload | mixed | Love the topic and the real datasets.`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | the lecture notes but finds grading inconsistent between TAs.`
- distilgpt2 + LoRA, merged: `workload | mixed | Liked the topic and the real datasets but found the weekly problem sets too much.`

**Comment:** Moodle is a nightmare. Links break, the quiz timer froze on me, and the audio on the recorded lectures cuts out every few minutes. I'm paying a lot for this.

- Reference: `technology | negative | Moodle has broken links and a frozen quiz timer, and recorded lecture audio cuts out repeatedly.`
- Keyword baseline (no model): `technology | positive | Moodle is a nightmare.`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | `
- distilgpt2 + LoRA, merged: `technology | mixed | Says the Moodle was a nightmare but wants more than one lecture on the recorded lectures.`

**Comment:** I came in worried about keeping up since I'd been away from school for ten years, but the way the first month built slowly from supply and demand up to the AD-AS model made it very manageable. Thank you for not skipping steps.

- Reference: `pacing | positive | Appreciates the gradual first-month build from supply and demand to AD-AS, which suited a returning student.`
- Keyword baseline (no model): `pacing | positive | I came in worried about keeping up since I'd been away from school for ten years, but the way the`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | `
- distilgpt2 + LoRA, merged: `pacing | mixed | Says the first month built on supply and demand but wants more than one AD-AS model.`

**Comment:** grading is soooo random. my friend wrote basically the same essay as me and got a B+ while I got a C. great fairness guys, 10/10

- Reference: `assessment | negative | Sarcastically criticises inconsistent essay grading, citing a near-identical essay scoring higher.`
- Keyword baseline (no model): `assessment | positive | grading is soooo random.`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | the book was a good book. I was very impressed with the book. I was very impressed with the book. I was very impressed with the book. I was very impressed with the book. I`
- distilgpt2 + LoRA, merged: `assessment | positive | Liked the random grading essay for getting a B+ in the course.`

**Comment:** Textbook explanations are really clear and the practice problems at the end of chapters are great, but the lecture slides on Canvas are just screenshots of the book with no added content, there's nothing to review from.

- Reference: `materials | mixed | Likes the clear textbook but says the slides are only book screenshots, offering nothing extra for revision.`
- Keyword baseline (no model): `materials | mixed | Textbook explanations are really clear and the practice problems at the end of chapters are great, but the lecture slides`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | `
- distilgpt2 + LoRA, merged: `materials | mixed | Praises clear textbook explanations but found the Canvas lecture slides incomplete.`

**Comment:** Heavy but fair. Clinical prep took about 8 hours a week and it was always clear why.

- Reference: `workload | positive | Considers the roughly eight weekly hours of clinical prep heavy but fair and purposeful.`
- Keyword baseline (no model): `workload | mixed | Heavy but fair.`
- distilgpt2, 3-shot prompt, no LoRA: `materials | mixed | Feels 18 hours a week is fair but says the expensive textbook was barely used.`
- distilgpt2 + LoRA, merged: `workload | mixed | Feels the clinical prep took about 8 hours a week and clear why.`

**Comment:** prof replies to emails on weekends too?? unreal. thanks

- Reference: `support | positive | Impressed the professor answers emails even on weekends.`
- Keyword baseline (no model): `support | negative | prof replies to emails on weekends too??`
- distilgpt2, 3-shot prompt, no LoRA: `unknown | unknown | `
- distilgpt2 + LoRA, merged: `support | negative | Says emails to email on weekends are unreal.`
