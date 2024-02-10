using System;
using System.Threading.Tasks;
using LLama;

namespace EdgeRag
{
    internal class ConversationManager
    {
        private IInputHandler inputHandler;
        private LLamaContext context;
        private InteractiveExecutor executor;
        private ChatSession session;
        private string[] prompts;
        private string[] antiPrompts;
        private int promptNumberChosen = 0;
    }
}
