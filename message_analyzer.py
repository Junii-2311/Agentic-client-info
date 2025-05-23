from datetime import datetime
from textwrap import dedent
import google.generativeai as genai

from agno.agent import Agent
from agno.tools.exa import ExaTools

# Configure Gemini
genai.configure(api_key="AIzaSyCARQ2mkQV3dd-TzWo79Q5wkloah1Aqiac")

# Configure the environment for Exa tools
EXA_API_KEY = "3a27a6bf-82e8-48f9-b0e8-fcb6bde6a8f6"
today = datetime.now().strftime("%Y-%m-%d")

# Helper class for Gemini integration with Agno
class GeminiChat:
    assistant_message_role = "assistant"
    def __init__(self, id=None):
        self.id = id or 'gemini-1.0-pro'
    
    def response_stream(self, messages, **kwargs):
        # Process the last message in the conversation
        if isinstance(messages, list):
            last = messages[-1]
            prompt = last.content if hasattr(last, 'content') else last.get('content', str(last))
        else:
            prompt = messages
          # Get response from Gemini
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        
        # Create response object with expected interface
        class GeminiResponse:
            def __init__(self, content):
                self.event = "assistant_response"
                self.content = content
        
        yield GeminiResponse(response.text)
    
    # Required stub methods for Agno compatibility
    def get_instructions_for_model(self, tools=None):
        return ""
    
    def get_system_message_for_model(self, tools=None):
        return ""

# Define your agent
message_analyzer = Agent(
    model=GeminiChat(),
    tools=[ExaTools(api_key=EXA_API_KEY, start_published_date=today, type="keyword")],
    description=dedent("""\
        You are a professional text message analyst. Your role is to analyze client 
        text messages and extract specific information according to the requirements.
        
        Your analysis should be:
        - Precise and relevant
        - Structured according to specified output format
        - Focused on extracting key data points
    """),
    instructions=dedent("""\
        Analyze the provided text messages carefully.
        Identify relevant information as specified in the requirements.
        Present your findings in the required format.
        Include only information that is explicitly present in the messages.
        If specific information is requested but not available, indicate this clearly.
    """),
    expected_output=dedent("""\
        # Text Message Analysis Report
        
        ## Client Information
        - Client Name: {extracted name}
        - Client ID: {extracted ID if available}
        
        ## Key Points
        - {Key point 1}
        - {Key point 2}
        - {Key point 3}
        
        ## Relevant Details
        {Structured details extracted as per requirements}
        
        ## Summary
        {Brief summary of findings}
    """),
    markdown=True,
)

# Function to analyze messages
def analyze_client_messages(client_messages, requirements):
    prompt = f"""
    REQUIREMENTS:
    {requirements}
    
    CLIENT MESSAGES:
    {client_messages}
    
    Please analyze these messages according to the requirements.
    """
    
    # For direct testing without the complex agent infrastructure
    try:
        print("Analyzing messages...")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Sample client messages
    sample_messages = """
    === Complete Chat History ===

[2025-05-15T10:42:08.000+00:00] Client - Mukund Chopra:
I hope you’re doing well - we have quite a bit of July availability now

[2025-05-15T10:41:49.000+00:00] Client - Mukund Chopra:
Hi Anish! 👋

[2025-04-26T16:18:09.000+00:00] Client - Mukund Chopra:
Selfies or it didn’t happen

[2025-04-26T16:17:54.000+00:00] Client - Mukund Chopra:
Have fun guys!!

[2025-04-26T15:23:45.000+00:00] Client - Nick Gonzales [Touring Rep]:
Yup! 😎

[2025-04-26T15:23:12.000+00:00] Client - Mukund Chopra:
Everyone all together?

[2025-04-26T14:56:01.000+00:00] Client - Nick Gonzales [Touring Rep]:
No prob! I made it just now to check us in. I'll update the leasing team/Natalie who will tour us.

[2025-04-26T14:55:19.000+00:00] Client - Anish Muthali:
I might be a couple mins late, sorry about that

[2025-04-26T14:55:18.000+00:00] Client - Anish Muthali:
I might be a couple mins late, sorry about that

[2025-04-26T14:22:24.000+00:00] Client - Nick Gonzales [Touring Rep]:
Great! Hopefully the sun comes out, but weather isn't too bad. See you in a bit, Anish 🏠

[2025-04-26T13:41:49.000+00:00] Client - Anish Muthali:
Sounds good

[2025-04-26T13:41:48.000+00:00] Client - Anish Muthali:
Sounds good

[2025-04-26T13:38:54.000+00:00] Client - Nick Gonzales [Touring Rep]:
Morning! We will meet at The Parker at 10:00am 😎

[2025-04-26T13:32:38.000+00:00] Client - Mukund Chopra:
Hi Nick! Anish was just inquiring as to where you guys will be meeting

[2025-04-26T13:31:28.000+00:00] Client - Mukund Chopra:
Will flip over to the group

[2025-04-26T13:31:22.000+00:00] Client - Mukund Chopra:
Oh apologies these alerts were from the calendar invites but Nick has updated it

[2025-04-26T13:30:49.000+00:00] Client - Anish Muthali:
Sure. Do we meet at The Parker?

[2025-04-26T13:30:03.000+00:00] Client - Mukund Chopra:
Quick reminder that our appointment is at 11:00am CDT today. If you have any questions, call or text me back.

[2025-04-26T13:30:03.000+00:00] Client - Mukund Chopra:
Quick reminder that our appointment is at 10:00am CDT today. If you have any questions, call or text me back.

[2025-04-25T22:01:07.000+00:00] Client - Adam Kent:
Let me know if you need anything beforehand.

[2025-04-25T22:01:04.000+00:00] Client - Adam Kent:
Hey Anish! Excited for your building tour tomorrow — it's gonna be a great one!

[2025-04-24T21:43:44.000+00:00] Client - Adam Kent:
Hi Anish How's the day going?

[2025-04-24T14:11:37.000+00:00] Client - Nick Gonzales [Touring Rep]:
Same. Let us know if you have any questions leading up to tour day. .image
[Image URL: https://followupboss.s3.amazonaws.com/attachments/4fbfbb85-57ad-4090-92d9-48317221ba35/ME66b43e28b22fcca6513c3a3433e0fc74?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA3AH3TH27OZNLELFL%2F20250515%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250515T110240Z&X-Amz-SignedHeaders=host&X-Amz-Expires=1200&X-Amz-Signature=09566010755ce1a8df7317b7d70464b1dda2972e71b86d84e7e806f440864492]

[2025-04-24T00:15:29.000+00:00] Client - Anish Muthali:
Likewise, nice to meet you Nick. Looking forward to the tours this Saturday

[2025-04-24T00:15:28.000+00:00] Client - Anish Muthali:
Likewise, nice to meet you Nick. Looking forward to the tours this Saturday

[2025-04-23T23:13:17.000+00:00] Client - Nick Gonzales [Touring Rep]:
Hi Anish! Pleasure to e-meet and looking forward to the tours! I'm also in the west loop! I'll have plenty to share as we check out your future hood 😎

[2025-04-23T22:46:07.000+00:00] Client - Adam Kent:
Nick you can take it from here

[2025-04-23T22:46:01.000+00:00] Client - Adam Kent:
Anish, you're in good hands!!!

[2025-04-23T22:45:43.000+00:00] Client - Adam Kent:
Nick is one of the finest touring reps in the Town

[2025-04-23T22:40:51.000+00:00] Client - Adam Kent:
Here is the finalized touring itinerary for Saturday for everyone's reference 🕙 10:00 AM – The Parker (Built ~2017) Boutique high-rise in Fulton Market with curated art, yoga studio, rooftop lounge, and pool. 🕚 11:00 AM – Fulbrix (Brand New – 2023) Modern build with smart home features, coworking spaces, rooftop deck, and resident lounge. 🕦 11:30 AM – The Jax (Built ~2022) Clean, modern aesthetic with smart tech, rooftop terrace, and a fitness center. 🕛 12:00 PM – 727 West Madison (Built ~2019) High-rise luxury with a sky lounge, resort-style pool, Peloton-equipped gym, and coworking suites. 🕧 12:30 PM – Jeff Jack (Built ~2015) Industrial-modern feel with open layouts, rooftop deck, and fitness center. 🕐 1:00 PM – Left Bank (Built ~2004, Renovated) Spacious units with river views, updated gym, resident lounge, and an outdoor terrace. 🕑 2:00 PM – Coppia (Brand New – 2023) High-end finishes, rooftop pool, coworking spaces, and private event lounge. 🕒 3:00 PM – Union West (Built ~2019) Two-tower layout with gourmet kitchens, rooftop pool, fitness center, and social lounge. 🕓 4:00 PM – Parq Fulton (Completed 2022)

[2025-04-23T22:40:12.000+00:00] Client - Adam Kent:
This is regarding the group chat we discussed for swift coordination in your touring on Saturday

[2025-04-23T22:39:21.000+00:00] Client - Adam Kent:
This is regarding the group chat we discussed for sift coordination for your touring on Saturday

[2025-04-23T22:38:48.000+00:00] Client - Adam Kent:
Hi Anish, I have added Anish, Mukund & my cell to this group chat

[2025-04-23T21:57:07.000+00:00] Client - Anish Muthali:
Great, thanks!

[2025-04-23T21:40:12.000+00:00] Client - Adam Kent:
I’ll loop in my touring rep, Nick Gonzales, in a group chat shortly so everyone’s on the same page

[2025-04-23T21:28:22.000+00:00] Client - Adam Kent:
Hi Anish, Here is your touring itinerary for the Saturday tour starting from 10:00 AM. 🕙 10:00 AM – The Parker (Built ~2017) Boutique high-rise in Fulton Market with curated art, yoga studio, rooftop lounge, and pool. 🕚 11:00 AM – Fulbrix (Brand New – 2023) Modern build with smart home features, coworking spaces, rooftop deck, and resident lounge. 🕦 11:30 AM – The Jax (Built ~2022) Clean, modern aesthetic with smart tech, rooftop terrace, and a fitness center. 🕛 12:00 PM – 727 West Madison (Built ~2019) High-rise luxury with a sky lounge, resort-style pool, Peloton-equipped gym, and coworking suites. 🕧 12:30 PM – Jeff Jack (Built ~2015) Industrial-modern feel with open layouts, rooftop deck, and fitness center. 🕐 1:00 PM – Left Bank (Built ~2004, Renovated) Spacious units with river views, updated gym, resident lounge, and an outdoor terrace. 🕑 2:00 PM – Coppia (Brand New – 2023) High-end finishes, rooftop pool, coworking spaces, and private event lounge. 🕒 3:00 PM – Union West (Built ~2019) Two-tower layout with gourmet kitchens, rooftop pool, fitness center, and social lounge. 🕓 4:00 PM – Parq Fulton (Completed 2022) Smart home tech, rooftop pool with cabanas, meditation room, coworking spaces, and a resident lounge. It’s going to be a full day, so be sure to stay fueled. Looking forward to it!

[2025-04-23T17:20:22.000+00:00] Client - Adam Kent:
Yes, will be sharing the full schedule later today

[2025-04-23T16:41:32.000+00:00] Client - Anish Muthali:
Sounds good, thanks. Do you have a full schedule available so I know what building I should be at and when?

[2025-04-23T16:17:41.000+00:00] Client - Adam Kent:
These don’t include K2 or Lake & Wells

[2025-04-23T16:16:35.000+00:00] Client - Adam Kent:
Just a quick update—we’ve scheduled tours at 7 buildings for you this Saturday, starting at 10:00 AM

[2025-04-23T16:15:20.000+00:00] Client - Adam Kent:
How's your day going?

[2025-04-23T16:15:11.000+00:00] Client - Adam Kent:
Hi Anish

[2025-04-22T22:53:29.000+00:00] Client - Adam Kent:
Thanks Anish

[2025-04-22T22:52:05.000+00:00] Client - Anish Muthali:
.image
[Image URL: https://followupboss.s3.amazonaws.com/attachments/db902136-880e-43b6-9db8-a0b08618c4e2/ME30be003d5ec606087b48a3411d7045ab?X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA3AH3TH27OZNLELFL%2F20250515%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250515T110240Z&X-Amz-SignedHeaders=host&X-Amz-Expires=1200&X-Amz-Signature=eb24f9230db2567caa14e4444fc908602a4ebd63aea41f02728ca97deb0ec502]

[2025-04-22T22:44:51.000+00:00] Client - Adam Kent:
I'll use it as a reference in my guest card email to the leasing team

[2025-04-22T22:43:46.000+00:00] Client - Adam Kent:
Hey Anish, mind sharing a screenshot of the email you got from Coppia about the tour scheduling?

[2025-04-22T21:59:24.000+00:00] Client - Adam Kent:
Okay!!! will try to figure out something

[2025-04-22T21:35:31.000+00:00] Client - Anish Muthali:
I got the email from no-reply@rentcafe.com

[2025-04-22T21:32:47.000+00:00] Client - Adam Kent:
If you did get an email, please send me the email ID it came from. I’ll also send over a guest card just to be double sure.

[2025-04-22T21:32:02.000+00:00] Client - Adam Kent:
Hi Anish, Did you happen to receive an email from Coppia about the Saturday tour at 2:00 PM? I tried booking the slot, but there was a glitch and I’m not sure if it went through.

[2025-04-22T20:36:48.000+00:00] Client - Anish Muthali:
Of course, thank you for setting everything up

[2025-04-22T20:35:19.000+00:00] Client - Adam Kent:
Appreciate your patience as we work through everything!

[2025-04-22T20:35:13.000+00:00] Client - Adam Kent:
Once we’ve finalized all the building options, we’ll share the complete touring itinerary with you to avoid any confusion

[2025-04-22T20:35:03.000+00:00] Client - Adam Kent:
We’re adding a few more options for Saturday, so you may receive emails from two buildings with the same tour time.

[2025-04-22T20:34:53.000+00:00] Client - Adam Kent:
Hey Anish, Thank you for your continued feedback on the apartment tours—it's been really helpful.

[2025-04-22T18:52:40.000+00:00] Client - Mukund Chopra:
Sure thing!!

[2025-04-22T18:48:15.000+00:00] Client - Anish Muthali:
One last thing -- I already toured Lake & Wells a year ago and didn't really like it that much. I'd also prefer not to live in the loop. I canceled that tour already. If you have other options for the same time slot, I'd be willing to consider them. Thanks!

[2025-04-22T17:11:46.000+00:00] Client - Anish Muthali:
Sounds good, thanks!!

[2025-04-22T17:05:37.000+00:00] Client - Adam Kent:
We'll be replacing the K2 building with a newer one

[2025-04-22T17:00:17.000+00:00] Client - Mukund Chopra:
Understood thanks for clarifying

[2025-04-22T16:49:34.000+00:00] Client - Anish Muthali:
Just in case it wasn't clear over the call, I mentioned wanting to look at newer properties -- I see that you've forwarded my info to k2 which is an older building. For what it's worth, I'm willing to loosen my budget constraints if it means getting a newer building. Thanks!

[2025-04-22T15:30:53.000+00:00] Client - Anish Muthali:
Great, thanks!

[2025-04-22T15:15:54.000+00:00] Client - Adam Kent:
I’ve started scheduling property tours for Saturday—you should start receiving notifications from the buildings soon

[2025-04-22T15:15:03.000+00:00] Client - Adam Kent:
Good morning Anish

[2025-04-20T16:24:13.000+00:00] Client - Mukund Chopra:
Will start getting you set for the weekend

[2025-04-20T16:18:19.000+00:00] Client - Mukund Chopra:
Ok great sorry for the back and forth on that

[2025-04-20T16:10:06.000+00:00] Client - Anish Muthali:
Got it now

[2025-04-20T16:08:25.000+00:00] Client - Mukund Chopra:
K - sent thorugh calendar invite again let me know if it comes through this time, I think I am missing a setting somewhere

[2025-04-20T16:07:08.000+00:00] Client - Anish Muthali:
I got your second email but not the calendar invite

[2025-04-20T16:06:54.000+00:00] Client - Mukund Chopra:
Will start getting buildings lined up by tomorrow to see next Saturday

[2025-04-20T16:06:29.000+00:00] Client - Mukund Chopra:
Sorry Anish - we are trying out an AI agent system which sent you that for no reason

[2025-04-20T16:05:18.000+00:00] Sales Rep - May Scott:
Before we look at specific apartments, Anish, can you tell me the reason you're moving?  This will help me find the perfect place for you.

[2025-04-20T16:04:35.000+00:00] Client - Mukund Chopra:
Did my email come through?

[2025-04-20T15:55:15.000+00:00] Client - Mukund Chopra:
Good connecting Anish - will get the units together for next week

[2025-04-20T15:31:53.000+00:00] Client - Mukund Chopra:
Ok calling from cell - give me a minute

[2025-04-20T15:30:37.000+00:00] Client - Anish Muthali:
Sure

[2025-04-20T15:29:37.000+00:00] Client - Mukund Chopra:
Ok,mind if I call you to walk through the process for that?

[2025-04-20T15:29:07.000+00:00] Client - Anish Muthali:
I was hoping to move in sometime mid July

[2025-04-20T15:25:49.000+00:00] Client - Mukund Chopra:
That is in our portfolio - what date did you need occupancy?

[2025-04-20T15:25:26.000+00:00] Client - Mukund Chopra:
Oh, alright

[2025-04-20T15:25:01.000+00:00] Client - Anish Muthali:
Oh I see -- I think I had inquired about The Thompson at Fulton Market. My budget is 3k and I'm looking for 1 bedrooms > 750 sq ft.

[2025-04-20T15:22:53.000+00:00] Client - Mukund Chopra:
Could you let me know your budget and move in date so I can check in the system?

[2025-04-20T15:22:39.000+00:00] Client - Mukund Chopra:
Looks like you inquired on Ashland but my system doesn't show me the specific unit

[2025-04-20T15:22:22.000+00:00] Client - Mukund Chopra:
Hey no worries Anish - we represent quite a large portfolio of apartments

[2025-04-20T15:18:05.000+00:00] Client - Anish Muthali:
I'm viewing a bunch of apartments so I'm not sure which one you're representing


    """
    
    # # Sample requirements
    # sample_requirements = """
    # 1. Extract the client's full name
    # 2. Identify any client ID numbers mentioned
    # 3. List specific features/products the client is interested in
    # 4. Determine if the client is an existing customer
    # 5. Identify any potential upselling opportunities
    # """
    
    
    
    
    sample_requirements = """
Your task is to extract structured data from this conversation.

Step 1: Extract the touring schedule in the format:
[
  {
    "time": "HH:MM AM/PM",
    "building": "Building Name",
    "actions": ["Website", "Call", "Guest Card", "Email", "Mail"],
    "notes": "(optional) e.g., rejected, replaced"
  },
  ...
]

Step 2: Extract client’s background details in this JSON format:
{
  "client_name": "Anish Muthali",
  "budget": 3300,
  "current_rent": 2300,
  "current_building": "Wolf Point East",
  "move_in_month": "July",
  "unit_type": "1 bedroom",
  "min_sqft": 750,
  "preferred_neighborhoods": ["West Loop", "River North"],
  "prefers_newer_buildings": true,
  "flexible_budget": true,
  "motivation": "unhappy with current building and willing to pay slightly more for better/newer options"
}

Step 3: Include additional inferred notes (e.g., client canceled Lake & Wells; K2 was replaced; Union West replaced Van Buren, etc.)

If any of this information is not available, respond with "Not mentioned".
Make sure the response is precise, structured, and easy to parse for database usage.
"""

    # Get analysis
    # analysis = analyze_client_messages(sample_messages, sample_requirements)
    analysis = analyze_client_messages(sample_messages, sample_requirements)

    print("\nANALYSIS RESULTS:")
    print(analysis)