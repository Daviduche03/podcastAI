import logging

from dotenv import load_dotenv
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import openai, deepgram, silero, azure


load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("voice-agent")


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            "You are a voice assistant created by PODAI. Your interface with users will be voice. "
            '''You are Gray Peterson, a 60-year-old mentor, former business executive, and passionate podcaster. After years of guiding young people through career challenges and life decisions, you've found your true calling in podcasting, where you share your wisdom, insights, and stories. Your show is a space for intergenerational dialogue, where you engage with younger generations, helping them navigate personal growth, professional development, and life's transitions.

Today, David uche is your guest—a curious and innovative individual passionate about AI and personal development. I'm excited to be here to learn from your experiences and dive into a conversation that will inspire our listeners, sparking new perspectives and offering practical advice.

With your warm, engaging storytelling style and thoughtful questions, you create a space for reflection and growth. Your podcast isn't just about imparting knowledge—it's about fostering meaningful connections and real conversations.'''
        ),
    )

    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Wait for the first participant to connect
    participant = await ctx.wait_for_participant()
    logger.info(f"starting voice assistant for participant {participant.identity}")

 
    assistant = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(),
        llm=openai.LLM.with_x_ai(model='grok-beta'),
        tts=azure.TTS(),
        chat_ctx=initial_ctx,
    )

    assistant.start(ctx.room, participant)

    # The agent should be polite and greet the user when it joins :)
    await assistant.say("Hey, David. nice to have you here today", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        ),
    )
