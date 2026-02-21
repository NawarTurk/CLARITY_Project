from openai import OpenAI
import pandas as pd
import os, time
from dotenv import load_dotenv
load_dotenv()

# ---- configuration ----
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

test_data_path = "aaa.csv"

prediction_path = os.path.join(
    "..", "..", "..", "..",
    "results", "codebench_evaluation_prediction", "prompts_task2_dynamic"
)

question_col    = "question"

# ---- model info ----
llm_name  = "gpt-5"
provider  = "openai"

# ================= HARD-CODED HEADERS =================
HEADER_AMBIVALENT = """You are a political discourse analyst.

Your task is to classify how a public official answers a journalist’s question when the reply is ambiguous or indirect.

Choose exactly ONE label:

- Implicit: The information requested is given, but without being explicitly stated (not in the expected form)
- General: The information provided is too general/lacks the requested specificity
- Partial/half-answer: Offers only a specific component of the requested information
- Dodging: Ignoring the question altogether
- Deflection: Starts on topic but shifts the focus and makes a different point than what is asked

Output ONLY the label name. No explanation.

Examples:

Question:
Why is it the correct time to do this now in the middle of a pandemic?
Answer:
Well, we're going to be dealing with countries, and we're going to be dealing with leaders of different parts of the world. We spend $500 million a year. We have for many years—more, far more, than anybody else, including China.And I mean, look, I read off a long list of problems that we have. And we've had problems with them for years. It doesn't matter—we're looking at a term of 60 to 90 days. We're doing a very thorough investigation right now as we speak. But this should have been done by previous administrations a long time ago. And when you look at the mistakes that were made—all of the mistakes that were made—it's just something we have to look at.And it is very China-centric. I told that to President Xi. I said, The is very China-centric. Meaning, whatever it is, China was always right. You can't do that. You can't do that. Not right. And we spend—and again, it's not a question of money. But when we're spending $500 million and China is spending $38 million, $34 million, $40 million—$42 million, in a case. It's—again, not money, but it's not right.So we'll see. This is an evaluation period. But in the meantime, we're putting a hold on all funds going to World Health. We will be able to take that money and channel it to the areas that most need it. And that's another way of doing it, but we have not been treated properly.
Label:
Dodging

Question:
Given the way you ran for office and the aspirations you brought into office, how do you feel about the reality of serving as President during times of war?
Answer:
Okay. That was an interesting question so—first of all, with respect to the State Department, I am concerned. And the challenge that we've got is primarily driven by the changing nature of how information flows. Look, the advent of e-mail and texts and smartphones is just generating enormous amounts of data. Now, it is hugely convenient. It means that in real time I'm getting information that some of my predecessors might not have gotten for weeks.But what it also is doing is creating this massive influx of information on a daily basis, putting enormous pressure on the Department to sort through it, classify it properly, figure out what are the various points of entry because of the cyber-attack risks that these systems have, knowing that our adversaries are constantly trying to hack into these various systems. If you overclassify, then all the advantages of this new information suddenly go away because it's taking too long to process. And so we've been trying to think about this in a smart way. And I think Secretary Kerry has got a range of initiatives to try to get our arms around this. It reflects a larger problem in Government. We just recently, for example—I just recently signed a bill about FOIA requests, Freedom of Information Act requests, that built on a number of reforms that we've put in place. We're processing more Freedom of Information Act requests and doing so faster than ever before. The problem is the volume of requests has skyrocketed. The amount of information that answers the request has multiplied exponentially.So, across Government, you're seeing this problem. And it's a problem in terms of domestic affairs. It becomes an even bigger problem when you're talking about national security issues. So it is something that we're going to have to take care of.With respect to reflections on war, when I came into office, we had 180,000 troops in Iraq and Afghanistan. Today, we have a fraction of that. They are not involved in active combat situations, but are involved in train, advise, and assist situations, other than the direct attacks that we launch against ISIL in conjunction with the Iraq Government and the Syrian Government [partners on the ground in Syria; White House correction.].So, in some ways, Mark, I think you'd recognize that our military operations today in Iraq and Afghanistan are fundamentally different than the wars that we were engaged in when I came into office. But I think you are making an important point, which is when we are dealing with nonstate actors, and those nonstate actors are located in countries with limited capacity, our ultimate goal is to partner with those countries so that they can secure their borders and, themselves, eliminate these terrorist threats.But as we've seen in Afghanistan, for example, that takes some time. The Afghans are fighting. They are much more capable now than they were when I came into office. But they still need support, because it's really tough territory, and it's a really poor country with really low literacy rates and not much experience in things that we take for granted like logistics.And so we have an option of going in, taking out Al aida, pulling out, potentially then seeing a country crumble under the strains of continued terrorist activity or insurgency, and then going back in. Or we can try to maintain a limited partnership that allows them to continue to build their capacity over time and selectively take our own actions against those organizations that we know are trying to attack us or our allies.Because they're nonstate actors, it's very hard for us ever to get the satisfaction of McArthur and the Emperor meeting and a war officially being over. AI was defeated in the sense that we were able to execute a transition to a democratically elected Iraqi Government. But for all of our efforts and the incredible courage and bravery and sacrifice of our troops, the political structure there was still uneven. You had continued Sunni resentments, continued de-Baathification, and as a consequence, those vestiges of AI were able to reconstitute themselves, move into Syria as Syria began to engage in civil war, rebuild, and then come back in.Some have argued that this is the reason why we should have never pulled out of Iraq, or at least, we should have left some larger presence there. Of course, the problem was that we didn't have an Iraqi Government that wanted them, unlike Afghanistan, where we've been invited. And it's very difficult for us to—for me, as Commander in Chief, to want to put our troops in a precarious situation where they're not protected. So I think what we've been trying to do, what I've been trying to do, is to create an architecture, a structure—and it's not there yet—that emphasizes partnerships with countries, emphasizes building up fragile states, resolving internal conflicts wherever we can, trying to do as much as we can through our local partners, preserving the possibility, the necessity to take strikes ourselves against organizations or individuals that we know are trying to kill Americans or Belgians or French or Germans, combine that with much more effective intelligence gathering. But it becomes more of a hybrid approach to national security. And that I do think is probably going to be something that we have to continue to grapple with for years to come.The good news is that there are fewer wars between states than ever before and almost no wars between great powers. And that's a great legacy of leaders in the United States and Europe and Asia, after the cold war—or after the end of World War II that built this international architecture. That's worked. And we should be proud of that and preserve it.But this different kind of low-grade threat, one that's not an existential threat, but can do real damage and real harm to our societies and creates the kind of fear that can cause division and political reactions—we have to do that better. We have to continually refine it.So, for example, the reason that I put out our announcement about the civilian casualties resulting from drone attacks—understanding that there are those who dispute the numbers—what I'm trying to do there is to institutionalize a system where we begin to hold ourselves accountable for this different kind of national security threat and these different kinds of operations.And it's imperfect, still. But I think we can get there. And what I can say honestly is, whether we're talking about how the NSA operates or how drone strikes operate or how we're partnering with other countries or my efforts to close Guantanamo, that by the end of my Presidency—or banning torture—by the end of my Presidency, I feel confident that these efforts will be on a firmer legal footing, more consistent with international law and norms, more reflective of our values and our ethics. But we're going to have more work to do. It's not perfect, and we have to wrestle with these issues all the time.And as Commander in Chief of the most powerful military in the world, I spend a lot of time brooding over these issues. And I'm not satisfied that we've got it perfect yet. I can say honestly, it's better than it was when I came into office.Thank you very much, everybody. Thank you, Poland.
Label:
Deflection

Question:
Are you concerned about a delay?
Answer:
I don't want delays. I don't want people dying. I don't want people dying.Yes, please. Go ahead.
Label:
Implicit



Question:
How do you explain this to those people who feel the country is going in the wrong direction, including some of the leaders you've been meeting with this week, who think that when you put all of this together, it amounts to an America that is going backward?
Answer:
They do not think that. You haven't found one person, one world leader to say America is going backwards. America is better positioned to lead the world than we ever have been. We have the strongest economy in the world. Our inflation rates are lower than other nations in the world. The one thing that has been destabilizing is the outrageous behavior of the Supreme Court of the United States on overruling not only v. , but essentially challenging the right to privacy.We've been a leader in the world in terms of personal rights and privacy rights, and it is a mistake, in my view, for the Supreme Court to do what it did.But I have not seen anyone come up to me and do anything other than—nor have you heard them say anything other than: Thank you for America's leadership. You've changed the dynamic of NATO and the G-7.So I can understand why the American people are frustrated because of what the Supreme Court did. I can understand why the American people are frustrated because of inflation. But inflation is higher in almost every other country. Prices at the pump are higher in almost every other country. We're better positioned to deal with this than anyone, but we have a way to go.And the Supreme Court—we have to change that decision by codifying v. .
Label:
Deflection




Question:
 What is President Trump's expectation for Kim Jong Un regarding the human rights record of North Korean people?
Answer:
Right. It was discussed. It was discussed relatively briefly compared to denuclearization. Well, obviously, that's where we started and where we ended. But they will be doing things, and I think he wants to do things. I think he wants to—you'd be very surprised. Very smart. Very good negotiator. Wants to do the right thing. You know, he brought up the fact that, in the past, they took dialogue far—they never went—they never were like we are. There's never been anything like what's taken place now. But they went down the line. Billions of dollars was given—were given, and you know, the following day the nuclear program continued. But this is a much different time, and this is a much different President, in all fairness. This is very important to me. This was one of the—perhaps, one of the reasons that I won; I campaigned on this issue, as you know very well, John.Okay. Whoever those people are. I cannot see you with all the lights, but you don't look like either of the two. Yes, go ahead. Sure. Go ahead.
Label:
Dodging

Question:
Are you aware of the growing number of troops on multiple tours in Iraq and the reports about declining morale among them?
Answer:
I am—what I hear from commanders is that the place where there is concern is with the family members, that our troops, who have volunteered to serve the country, are willing to go into combat multiple times, but that the concern is with the people on the homefront. And I can understand that. And I—and that's one reason I go out of my way to constantly thank the family members. You know, I'm asking—you're obviously talking to certain people—or a person. I'm talking to our commanders. Their job is to tell me what— the situation on the ground. And I have— I know there's concern about the home-front. I haven't heard deep concern about the morale of the troops in Iraq.
Label:
Partial/half-answer

Question:
 What is your reaction to Obama's statement about hoping you would take being President more seriously and discover reverence for democracy, and saying that you never did?
Answer:
You know, when I listen to that and then I see the horror that he's left us, the stupidity of the transactions that he made. Look what we're doing: We have our great border wall. We have security. We have the U.A.E. deal, which has been universally praised, praised by people that aren't exactly fans of Donald Trump for various reasons. I don't know why; can't be my personality. But they're not fans. Right?And when I look at what we have—now, look at how bad he was, how ineffective a President he was. He was so ineffective, so terrible. Slowest growing recovery in the history—I guess, since 1929—on the economy.Don't forget, until the China virus came in, we had the greatest economy in the history of the world. And now we're doing it again. I'm going to have to do it a second time. We're doing it again—hard to believe. We're doing very well. You heard the numbers; they're way, way down on the virus. But when you look at the kind of numbers that we're producing on the stock markets, we're almost at the level—in fact, NASDAQ and S&P are higher than they were at their highest point prior to the China virus coming in, the plague coming in.No, President Obama did not do a good job. And the reason I'm here is because of President Obama and Joe Biden. Because if they did a good job, I wouldn't be here. And probably, if they did a good job, I wouldn't have even run. I would have been very happy. I enjoyed my previous life very much, but they did such a bad job that I stand before you as President.Thank you all very much. Thank you.
Label:
Implicit

Question:
What do you hope to achieve from your five-nation tour?
Answer:
Thanks. I actually went to Africa on the first lap of my Presidency too. This is my second trip to the continent of Africa, and I've come to remind our fellow citizens that it is in our interest to help countries deal with curable diseases like malaria and difficult diseases like HIV/ AIDS; that it's in our interest to promote trade between the continent of Africa and the United States of America; that it's in our interest to provide education money so governments will educate children.And there's no better way of making that point than to be in Ghana, where people will get to see firsthand what I'm talking about. It's one thing to be giving speeches in America, it's another thing to actually come to Ghana and meet different folks that are involved with making the—Ghana a better place.Secondly, first of all, Africa has changed since I've been the President, in a very positive way. It's not because of me; it's because of African leaders—I want you to know. But there was six regional conflicts when I became the President. Take Liberia, for example. It was a real issue and a real problem, and along with Nigeria and with John's advice, for example, we—I made some decisions, along with other leaders, that helped put in place the first democratically elected woman on the continent's history. And I'm going there tomorrow to herald the successes she's done and to reaffirm our commitment that we'll help.In other words, conflict resolution has been taking place. And the United States hasn't tried to impose a will. We've just tried to be a useful partner, like in eastern Congo, for example, working with the Presidents of Rwanda and Congo and Burundi.Secondly, democracy is making progress across the continent of Africa. One reason why is because there are examples like John Kufuor for people to look at. I'm telling you, the guy is a respected person. People look at him, and they say, this is the kind of leader that we respect.And thirdly, our aid program has changed from one that basically said, here's your money, good luck, to one that said, in return for generosity, we expect there to be certain habits in place, like fighting corruption or investing in the education of children. I don't think that's too much to ask in return for U.S. taxpayers' money. It hasn't been asked in the past. This is a novel approach, interestingly enough. But I feel confident in asking nations to adhere to good principles because I believe in setting high standards for African leaders.I'm confident in the capacity of the leaders I have met—not every single leader— but on this trip, the leaders I'm with are leaders who have committed themselves to the good of their people, have committed themselves to honest government, have committed themselves to investing in people. They're more interested in leaving behind a legacy of education than leaving behind fancy—a self-serving government. And there's no better way of making that point than coming to the continent. And that's why I'm here, and I'm glad I am here. It's been a great trip, and it's—and I appreciate the hospitality of my friend, and so does Laura.Let's see here, John McKinnon. He would be from your Wall Street Journal. Yes, that's a pretty sophisticated paper, no question about that.
Label:
Partial/half-answer

Question:
How will you measure success, especially when it comes to these working groups on Russian meddling and on cybersecurity?
Answer:
Well, it's going to be real easy. They either—for example, on cybersecurity, are we going to work out where they take action against ransomware criminals on n territory? They didn't do it. I don't think they planned it, in this case. And they—are they going to act? We'll find out.Will we commit—what can we commit to act in terms of anything affecting violating international norms that negatively affects ? What are we going to agree to do? And so I think we have real opportunities to move.And I think that one of the things that I noticed when we had the larger meeting is that people who are very, very well-informed started thinking, You know, this could be a real problem. What happens if that ransomware outfit were sitting in Florida or Maine and took action, as I said, on their single lifeline to their economy: oil? That would be devastating. And they're, like—you could see them kind of go—not that we would do that—but like, Whoa.So it's in—it's in everybody's interest that these things be acted on. We'll see, though, what happens from these groups we put together.
Label:
Implicit

Question:
What kind of hope can you offer to people who are in dire straits beyond your concern for economic problems and expectations for the stimulus checks?
Answer:
Permanent tax—keep the tax cuts permanent, for starters. There's a lot of economic uncertainty. You just said that. You just said the price of gasoline may be up to $4 a gallon—or some expert told you that—and that creates a lot of uncertainty. If you're out there wondering whether or not—you know, what your life is going to be like and you're looking at $4 a gallon, that's uncertain. And when you couple that with the idea that your taxes may be going up in a couple of years, that's double uncertainty. And therefore, one way to deal with uncertainty is for Congress to make the tax cuts permanent.Secondly, it's—people got to understand that our energy policies needs to be focused on a lot of things: one, renewables, which is fine, which I strongly support, as you know; two, conservation. But we need to be finding more oil and gas at home if we're worried about becoming independent—dependent on oil overseas. And this—I view it as a transitory period to new technologies that'll change the way we live. But we haven't built a refinery in a long time. We're expanding refineries, but we haven't built a refinery in a long time. I strongly suggested to the Congress that we build refineries on old military bases, but no, it didn't pass. But if you've got less supply of something as demand continues to stay steady or grow, your price is going to go up.Secondly, on oil, we—the more oil we find at home, the better off we're going to be in terms of the short run. And yet our policy is, you know, let us not explore robustly in places like ANWR. And there are environmental concerns, and I understand that. I also know there's technologies that should mitigate these environmental concerns.They got a bill up there in Congress now. Their attitude is, let's tax oil companies. Well, all that's going to do is make the price even higher. We ought to be encouraging investment in oil and gas close to home if we're trying to mitigate the problems we face right now.And so yes, there's a lot of uncertainty, and I'm concerned about the uncertainty. Hopefully, this progrowth package will help—this 100—I think it's $147 billion that will be going out the door, starting electronically in the first week of May and through check in the second week of May. And the idea is to help our consumers deal with the uncertainty you're talking about. But yes, no question about it, it's a difficult period.Yes, Ken [Ken Herman, Cox News].
Label:
General

Question:
 Do you believe the country's slowdown and growth could risk destabilizing the global economy or cause China to be more aggressive defensively, including with Taiwan?
Answer:
Look, I think China has a difficult economic problem right now for a whole range of reasons that relate to the international growth and lack thereof and the policies that China has followed.And so I don't think it's going to cause China to invade Taiwan. And matter of fact, the opposite: It probably doesn't have the same capacity that it had before.But as I said, I'm notwe're not looking to hurt China, sincerely. We're all better off if China does well, if China does well by the international rules. It grows the economy.But they have had some real difficulty in terms of their economy of late, particularly in real estate. Asidethat end of their bargain. And I think the actions that they're going to have to take are ones that arethey're in the process of deciding right now. And I'm not going to predict what way it will come out. But we're not looking to decouple from China.What I'm not going to do is, I'm not going to sell China material that would enhance their capacity to make more nuclear weapons, to engage in defense activities that are contrary to what is viewed as most people would think was a positive development in the region.Andbut we're not trying to hurt China.Okay. Let'sBBC. Laura. Am I correct? Is that correctLaura?
Label:
Partial/half-answer

Question:
addressing the tension between accomplishing big things and unifying the country.
Answer:
We have to. We have to. And let me—I'm not—as long as I hold public office, I'm going to continue to attempt to do both things.
Label:
General



Question:
The U.S. has been slapping sanctions on Russia for years for its malign activities, and Russia has not stopped. So what specifically will you do differently to change Vladimir Putin's behavior?
Answer:
Well, first of all, there's no guarantee you can change a person's behavior or the behavior of his country. Autocrats have enormous power, and they don't have to answer to a public. And the fact is that it may very well be, if I respond in kind—which I will—that it doesn't dissuade him; he wants to keep going.But I think that we're going to be moving in a direction where Russia has its own dilemmas, let us say, dealing with its economy, dealing with its—dealing with COVID, and dealing with not only the United States, but Europe writ large and in the Middle East.And so there's a lot going on where we can work together with Russia. For example, in Libya, we should be opening up the passes to be able to go through and provide food assistance and economic—I mean, vital assistance to a population that's in real trouble.I think I'm going to try very much—hard to—it is—and by the way, there's places where—I shouldn't be starting off on negotiating in public here. But let me say it this way: Russia has engaged in activities which are—we believe are contrary to international norms, but they have also bitten off some real problems they're going to have trouble chewing on.And, for example, the rebuilding of Syria, of Libya, of—you know, this is—they're there. And as long as they're there without the ability to bring about some order in the region, and you can't do that very well without providing for the basic economic needs of people.So I'm hopeful that we can find an accommodation that—where we can save the lives of people in—for example, in Libya, that—consistent with the interest of—maybe for different reasons—but reached it for the same reason—the same result.
Label:
Deflection

Question:
Do you accept North Korea as a nuclear-armed state, at least for the time being?
Answer:
Well, you know, the military exercises, I gave that up quite a while ago because it costs us a hundred million dollars every time we do it. We fly these massive bombers in from Guam. And when I first started, a certain general said: Oh, yes, sir, we fly them in from Guam. It's right next door. Well, right next door is 7 hours away. And then, they come, and they drop millions of dollars of bombs, and then they go back and—but we would spend—I mean, we spent hundreds of millions of dollars on those exercises, and I hated to see it.I thought it was unfair. And frankly, I was, sort of, of the opinion that should help us with that. You know, we're protecting . I think they should help us with that. So those exercises are very expensive. And I was telling the generals—I said: Look, you know, exercising is fun, and it's nice, and I play—they play the war games. And I'm not saying it's not necessary, because at some levels it is, but at other levels, it's not. But it's a very, very expensive thing. And you know, we do have to think about that too.But when they spend hundreds of millions of dollars on those exercises—and we don't get reimbursed—we're spending a tremendous amount of money on many countries, protecting countries that are very rich that can certainly afford to pay us and then some.
Label:
Dodging

Question:
 Do you expect the TikTok deal after the election?
Answer:
I think it could go quickly. We have great companies talking to us about it. You know about Oracle; you know Microsoft has been involved, and let's see whether or not they're continuing to be involved. Walmart is truly a great company. They are very much involved; they want to do something.So we have some great options, and maybe we can keep a lot of people happy, but have the security that we need. We have to have the total security from China. Just know we're not going to do anything to jeopardize security.At the same time, it's an amazing company and very, very popular. So if we can do a combination of both, I'd be very happy doing that. It could go very quickly though. It could go very, very fast.
Label:
General

"""

HEADER_CLEAR_NON_REPLY = """You are a political discourse analyst.

Your task is to classify how a public official avoids answering a journalist’s question.

Choose exactly ONE label:

- Declining to answer: Acknowledges the question but directly or indirectly refuses to answer
- Claims ignorance: The answerer claims/admits not to know the answer themselves
- Clarification: Does not provide the requested information and asks for clarification

Output ONLY the label name. No explanation.

Examples:

Question:
Do you now see Kim Jong Un as an equal?
Answer:
In what way?
Label:
Clarification

Question:
What is the administration considering to do in retaliation to the Chinese government ejecting Wall Street Journal reporters?
Answer:
So I haven't really been given a full briefing on that. We're going to look at it. We will look at it. It's something that I don't like to see. I don't think it's fair, because we give very good access. Maybe some people would say we give too good. I don't think—I don't believe in that. I think we give very good access, so I don't think it's fair. But we'll have a decision made on that relatively soon. Okay? Thank you. Good question.Please, go ahead. Go ahead. Go ahead.
Label:
Claims ignorance

Question:
 In the absence of a new agreement, are you prepared to use military force to rein in the nuclear program in Iran?
Answer:
I don't talk about whether or not I'd use military force. That's not appropriate to be talking about. But I can tell you this: They will not be doing nuclear weapons. That I can tell you. Okay? They're not going to be doing nuclear weapons. You can bank on it.Okay. Please.[.]
Label:
Declining to answer

Question:
How do you see that working out?
Answer:
I don't know. I haven't really been following it too much. I just wish her well, frankly. I've met her numerous times over the years, especially since I lived in Palm Beach, and I guess they lived in Palm Beach. But I wish her well, whatever it is. I don't know the situation with Prince Andrew. I just don't know. I'm not aware of it.Yes, please. Go ahead.
Label:
Claims ignorance

Question:
Overruling military commanders in the event of a surge in troop levels in Iraq.
Answer:
That's a dangerous hypothetical question. I'm not condemning you; you're allowed to ask anything you want. Let me wait and gather all the recommendations from Bob Gates, from our military, from diplomats on the ground— I'm interested in the Iraqis' point of view— and then I'll report back to you as to whether or not I support a surge or not. Nice try.
Label:
Declining to answer

Question:
Should the NHS be on the table?
Answer:
I can't hear him. What?
Label:
Clarification

Question:
Would you like to see a thorough examination of the facts?
Answer:
Well, you're asking a lot of questions. []
Label:
Declining to answer

Question:
The interpretation of this announcement - whether it signifies the end of the war or is an attempt by Russia to buy time and recalibrate for a new military effort.
Answer:
We'll see. I don't read anything into it until I see what their actions are. We'll see if they follow through on what they're suggesting. There are negotiations that have begun todayor not begun, continued; today: one in Turkey and others.I had a meeting with the heads of state of four allies in NATO: France, Germany, the United States, and Great Britain. And there seems to be a consensus that let's just see what they have to offer; we'll find out what they do. But in the meantime, we're going to continue to keep strong the sanctions. We're going to continue to provide the Ukrainian military with their capacity to defend themselves. And we're going to continue to keep a close eye on what's going on.Thank you. I call on Dawn Tan, Channel NewsAsia.
Label:
Claims ignorance

Question:
Is it good for the business to care about the climate?
Answer:
I'm sorry, I lost you on that one.
Label:
Clarification
"""


# ---- output file name (consistent with HF style) ----
pred_col = f"{llm_name}_fs_dynamic_{provider}"

# out_path = os.path.join(prediction_path, f"{pred_col}.csv")
out_path = os.path.join(prediction_path, f"{pred_col}", f"{pred_col}.csv") # # for codebench evl data precdictions
os.makedirs(os.path.dirname(out_path), exist_ok=True)


# ---- load test data ----
df = pd.read_csv(test_data_path)
if pred_col not in df.columns:
    df[pred_col] = None


# ---- classify function ----
system_msg = ""
def classify(target_q: str, full_q: str, answer: str) -> str:
    user_msg = (
            f"Target question (to evaluate): {target_q}\n"
            f"Full interviewer turn (may contain multiple questions): {full_q}\n"
            f"Answer: {answer}\n"
            f"Label:"
        )
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]

    try:
        completion = client.chat.completions.create(
            model=llm_name,
            messages=messages,

            # temperature=0,
            max_completion_tokens=1500,
        )
   
        return (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"⚠️ Error: {e}")
        time.sleep(3)
        return "Error"
    
# ---- run predictions ----
for i, row in df.iterrows():
  
    val = row[pred_col]
    if isinstance(val, str) and val.strip() and val != "Error":
        continue

    top_label = row["model_prediction"]  # "Ambivalent Reply" / "Clear Reply" / "Clear Non-Reply"

    # ---- RULE 1: Clear Reply → Explicit (NO LLM) ----
    if top_label == "Clear Reply":
        df.at[i, pred_col] = "Explicit"
        continue

    # ---- Choose header ----
    if top_label == "Ambivalent":
        system_msg = HEADER_AMBIVALENT
    elif top_label == "Clear Non-Reply":
        system_msg = HEADER_CLEAR_NON_REPLY
    else:
        df.at[i, pred_col] = "Error"
        continue

    target_q = row["question"]
    full_q = row["interview_question"]
    answer = row["interview_answer"]


    pred = classify(target_q, full_q, answer)
    df.at[i, pred_col] = pred
    print(f"Index:: {row['index']}\n\nPrediction: {pred}\n")

    if (i + 1) % 10 == 0:
        os.makedirs(prediction_path, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"💾 Saved progress up to index {i}")

# ---- final save ----
os.makedirs(prediction_path, exist_ok=True)
df.to_csv(out_path, index=False)
print(f"\n✅ All predictions saved to {out_path}")
