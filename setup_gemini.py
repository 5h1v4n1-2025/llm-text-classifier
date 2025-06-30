#!/usr/bin/env python3
"""
Helper script to set up Gemini API key for the text classifier
"""

import os
import getpass

def setup_gemini_api():
    """
    Set up Gemini API key
    """
    print("🔑 Gemini API Key Setup")
    print("=" * 50)
    
    # Check if API key already exists
    existing_key = os.getenv('GEMINI_API_KEY')
    if existing_key:
        print(f"✅ Gemini API key already found in environment")
        print(f"   Key: {existing_key[:10]}...{existing_key[-4:]}")
        return True
    
    print("To use Gemini LLM for text classification, you need a Gemini API key.")
    print("\nHow to get your API key:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the generated key")
    
    print("\n" + "=" * 50)
    
    # Get API key from user
    api_key = getpass.getpass("Enter your Gemini API key (input will be hidden): ").strip()
    
    if not api_key:
        print("❌ No API key provided. The script will use keyword-based classification.")
        return False
    
    # Test the API key
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content("Hello")
        print("✅ API key is valid!")
        
        # Save to environment (for current session)
        os.environ['GEMINI_API_KEY'] = api_key
        
        # Ask if user wants to save permanently
        save_permanent = input("\nDo you want to save this API key permanently? (y/n): ").lower().strip()
        if save_permanent in ['y', 'yes']:
            # Add to shell profile
            shell_profile = os.path.expanduser("~/.zshrc")  # For zsh
            if not os.path.exists(shell_profile):
                shell_profile = os.path.expanduser("~/.bash_profile")  # Fallback to bash
            
            with open(shell_profile, 'a') as f:
                f.write(f'\n# Gemini API Key for text classifier\nexport GEMINI_API_KEY="{api_key}"\n')
            
            print(f"✅ API key saved to {shell_profile}")
            print("   You'll need to restart your terminal or run 'source ~/.zshrc' for the changes to take effect.")
        
        return True
        
    except Exception as e:
        print(f"❌ Invalid API key or API error: {e}")
        print("The script will use keyword-based classification instead.")
        return False

if __name__ == "__main__":
    setup_gemini_api() 