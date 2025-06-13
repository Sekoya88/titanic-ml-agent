#!/usr/bin/env python3
"""
Training script for Titanic ML Agent
Quick execution script following the roadmap
"""

from titanic_agent import TitanicModelAgent

def main():
    print("🚀 Titanic ML Agent - Training Started!")
    print("=" * 50)
    
    # Initialize and run the agent
    agent = TitanicModelAgent()
    agent.load_data()
    agent.quick_eda()
    agent.train_model()
    agent.predict_and_save()
    agent.save_model()
    
    print("=" * 50)
    print("🎉 Training completed! Check submission.csv for results.")

if __name__ == "__main__":
    main() 