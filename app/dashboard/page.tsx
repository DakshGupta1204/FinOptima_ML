"use client";
import { useState, useEffect } from "react";
import { LogOut, User, Sun, Moon, BarChart3, PieChart, LineChart, Briefcase, X } from "lucide-react";
import { useRouter } from "next/navigation";

const Dashboard = () => {
  const [darkMode, setDarkMode] = useState(false);
  const [selectedTool, setSelectedTool] = useState<{ name: string; desc: string; link: string } | null>(null);
  const router = useRouter();
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [darkMode]);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };
  const token = localStorage.getItem("token");
  const isLoggedIn = token ? true : false;

  const tools = [
    {
      name: "Risk Assessment",
      icon: BarChart3,
      desc: "The Risk Assessment Engine is an intelligent system designed to evaluate and categorize individual financial risk profiles. By analyzing key user inputs, it provides a personalized risk category—Conservative, Moderate, or Aggressive—along with a confidence score that reflects the reliability of the assessment. Additionally, it generates a risk score and an adjusted market score, taking into account broader financial conditions.This model enables users to make informed financial decisions by offering tailored insights into their risk-taking tendencies. It ensures accuracy, transparency, and adaptability to market changes, helping users align their investment strategies with their personal financial outlook.",
      link: "/ml-risk",
    },
    {
      name: "Portfolio Management",
      icon: PieChart,
      desc: "The Portfolio Optimizer is a powerful tool designed to help investors construct an optimal investment portfolio based on risk preferences and constraints. It provides:Optimized Portfolio Allocation – A balanced distribution of investments across selected assets.Risk & Performance Metrics – Key insights such as expected returns, volatility, Sharpe ratio, and risk exposure.Customizable Constraints – Options to set asset weight limits, sector allocations, and turnover controls.Visualization & Reports – Clear charts and reports for informed decision-making.This tool enables users to make data-driven investment decisions, ensuring a well-diversified and risk-managed portfolio tailored to their needs.",
      link: "/ml-portfolio",
    },
    {
      name: "Market Analysis",
      icon: LineChart,
      desc: "The **Market Sentiment Analyzer** provides real-time insights into financial market trends by analyzing news sentiment from multiple sources. It delivers an overall market sentiment score, a detailed breakdown of financial sentiment (positive, negative, or neutral), and a general sentiment analysis using advanced NLP models. By aggregating data from top financial news providers, it offers comprehensive reports with relevant articles, helping investors and businesses make informed decisions with a clear understanding of market sentiment.",
      link: "/ml-market",
    },
    {
      name: "Life Events Planning",
      icon: Briefcase,
      desc: "This AI-powered financial assistant helps individuals make smarter financial decisions by analyzing income, expenses, debt, and investments while simulating life events like career shifts or retirement. It provides **personalized debt management**, **optimized investment strategies**, and **risk-aware financial planning** using advanced AI techniques like Markov Decision Processes. Whether you're repaying loans, growing your wealth, or preparing for uncertainties, the system offers **data-driven recommendations** to **maximize returns, minimize risks, and ensure long-term financial stability**—empowering you to make confident financial choices. 🚀",
      link: "/ml-investment",
    },
  ];

  return (
    <div className="w-full min-h-screen bg-gray-100 dark:bg-[#121212] transition-all duration-300 flex flex-col">
      {/* Navbar */}
      <nav className="flex justify-between items-center p-16 mb-8">
        <div className="flex items-center space-x-2">
          <LineChart className="w-8 h-8 text-primary dark:text-primary-light" />
          <span className="text-2xl font-bold text-primary-dark dark:text-white">FinOptima</span>
        </div>
        <div className="flex items-center gap-4">
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-primary dark:text-primary-light hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
          <button className="bg-primary hover:bg-primary-dark text-white px-6 py-2 rounded-lg transition-colors" onClick={()=>router.push(isLoggedIn?"/profile":"/auth")}>
            {isLoggedIn?"Profile":"Login"}
          </button>
          {isLoggedIn && <button className="bg-primary hover:bg-red-500 text-white px-6 py-2 rounded-lg transition-colors" onClick={() => localStorage.removeItem("token")}>
            Logout
          </button>}
        </div>
      </nav>

      {/* Content Section */}
      <div className="container mx-auto flex flex-col items-center mt-12 px-8">
        <h1 className="text-5xl font-bold text-gray-900 dark:text-white mb-8">
          Welcome to Your Dashboard
        </h1>
        <p className="text-xl text-gray-700 dark:text-gray-300 mb-10 max-w-3xl text-center">
        Manage your financial decisions effectively with our advanced tools. Each module is designed to help you plan, analyze, and optimize your investments.
        </p>

        {/* Tool Modules */}
        <div className="w-full grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {tools.map(({ name, icon: Icon, desc, link }, index) => (
            <div
              key={index}
              onClick={() => setSelectedTool({ name, desc, link })}
              className="relative p-8 bg-white dark:bg-primary-light shadow-xl rounded-2xl border border-primary dark:border-primary-dark transition transform hover:scale-105 cursor-pointer flex flex-col items-center justify-center gap-6 hover:shadow-2xl"
            >
              <Icon className="w-20 h-20 text-primary dark:text-primary transition-transform duration-300 transform hover:scale-125" />
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-black">{name}</h2>
            </div>
          ))}
        </div>
      </div>

      {/* Modal */}
      {selectedTool && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-50">
          <div className="bg-white dark:bg-[#1e1e1e] p-6 rounded-lg w-[40%] shadow-lg relative">
            <button onClick={() => setSelectedTool(null)} className="absolute top-3 right-3">
              <X className="w-6 h-6 text-gray-600 dark:text-gray-300" />
            </button>
            <h2 className="text-3xl font-semibold text-gray-900 dark:text-white mb-6">
              {selectedTool.name}
            </h2>
            <p className="text-1xl text-gray-700 dark:text-gray-300 mb-6">{selectedTool.desc}</p>
            <a
              href={selectedTool.link}
              className="bg-primary hover:bg-primary-dark text-white px-6 py-2 rounded-lg transition-colors inline-block"
            >
              Go to {selectedTool.name}
            </a>
          </div>
        </div>
      )}
    </div>
  );
};

export default Dashboard;
