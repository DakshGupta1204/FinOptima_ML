"use client";
import React, { useState, useEffect } from "react";
import axios from "axios";
import {
  User,
  LogOut,
  LineChart,
  Calculator,
  Bot,
  ShieldCheck,
  PiggyBank,
  Moon,
  Sun,
  Mail,
  ChartBar,
  Wallet,
  Target,
  Brain,
  Shield,
  Lock,
  ChevronDown,
  Loader2,
  TrendingUp,
  BarChart,
  TrendingDown,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useRouter } from "next/navigation";

const RiskAssessment: React.FC = () => {
  const router = useRouter();
  const [loading, setLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false); // Theme State

  interface Sentiment {
    overall_sentiment:{
      financial_sentiment: {
        positive: number;
        neutral: number;
        negative: number;
      };
    }
  }

  const [sentiment, setSentiment] = useState<Sentiment | null>(null);

  const [newsArticles, setNewsArticles] = useState<any[]>([]);

  const fetchSentiment = async () => {
    try {
      setLoading(true);
      const response = await axios.get("http://127.0.0.1:8000/market-sentiment"); // Adjust API URL
      console.log("REsponse: ",response.data);
      setSentiment(response.data);
      setNewsArticles(response.data.analyses);
    } catch (error) {
      console.error("Error fetching sentiment:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchSentiment();
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
    if (!darkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const backgroundIcons = [
    { Icon: Mail, size: 200, top: 60, left: 40, rotate: "rotate-12" },
    { Icon: Lock, size: 150, bottom: 40, right: 60, rotate: "-rotate-12" },
    { Icon: ChartBar, size: 180, top: "7%", left: "40%", rotate: "rotate-135" },
    { Icon: Wallet, size: 160, top: "30%", right: "1%", rotate: "-rotate-15" },
    { Icon: PiggyBank, size: 140, bottom: "1%", left: "20%", rotate: "rotate-30" },
    { Icon: LineChart, size: 170, top: "2%", right: "18%", rotate: "-rotate-20" },
    { Icon: Target, size: 130, bottom: "30%", right: "25%", rotate: "rotate-25" },
    { Icon: Brain, size: 150, top: "60%", left: "1%", rotate: "-rotate-35" },
    { Icon: Shield, size: 120, bottom: "5%", left: "55%", rotate: "rotate-15" },
  ];

  const [dropdownOpen, setDropdownOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gray-100 text-gray-900 dark:bg-[#0d1117] dark:text-white z-0">
      <div className="absolute inset-0 opacity-20 dark:opacity-10 ">
        {backgroundIcons.map(({ Icon, size, ...style }, idx) => (
          <Icon
            key={idx}
            size={size}
            className={`text-primary-dark dark:text-primary-light absolute ${style.rotate} transition-all duration-500`}
            style={{
              top: style.top,
              left: style.left,
              bottom: style.bottom,
              right: style.right,
            }}
          />
        ))}
      </div>
      {/* Navbar */}
      <nav className="flex justify-between items-center p-16 z-20 relative">
        <h1 className="text-2xl font-bold flex items-center justify-center gap-2 text-primary dark:text-white cursor-pointer" onClick={() => router.push("/")}>
          <LineChart size={32} className="dark:text-primary-light" /> FinOptima
        </h1>

        {/* ML Tools in Navbar */}

        <div className="flex items-center space-x-4">
          {/* Theme Toggle */}
          <div className="relative">
            <button
              onClick={() => setDropdownOpen(!dropdownOpen)}
              className="flex items-center bg-gray-200 dark:bg-gray-800 px-3 py-2 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition"
            >
              <Bot size={20} className="mr-2" />
              ML Tools
              <ChevronDown size={18} className="ml-2" />
            </button>

            {dropdownOpen && (
              <div className="absolute right-0 mt-2 w-56 bg-white dark:bg-gray-900 shadow-md rounded-md py-2 z-50">
                {[
                  { name: "Risk Assessment", icon: LineChart, link: "/ml-risk" },
                  { name: "Portfolio Optimizer", icon: Calculator, link: "/ml-portfolio" },
                  { name: "Market Analysis", icon: Bot, link: "/ml-market" },
                  { name: "Financial Planning", icon: ShieldCheck, link: "/ml-financial" },
                  { name: "Investment Planner", icon: PiggyBank, link: "/ml-investment" },
                ].map(({ name, icon: Icon, link }, idx) => (
                  <button
                    key={idx}
                    className="flex items-center px-4 py-2 w-full text-left hover:bg-gray-100 dark:hover:bg-gray-700 transition"
                    onClick={() => router.push(link)}
                  >
                    <Icon size={18} className="mr-3" />
                    {name}
                  </button>
                ))}
              </div>
            )}
          </div>
          <button
            onClick={toggleDarkMode}
            className="p-2 rounded-md hover:bg-gray-300 transition bg-gray-200 dark:bg-gray-800 dark:hover:bg-gray-600 text-primary dark:text-primary-light"
          >
            {darkMode ? <Sun size={22} /> : <Moon size={22} />}
          </button>

          {/* Profile & Logout */}
          <button className="flex items-center space-x-1 bg-primary px-3 py-2 rounded-lg hover:bg-primary-dark transition text-white" onClick={() => router.push("/user-profile")}>
            <User size={20} />
            <span>Profile</span>
          </button>
          <button className="flex items-center space-x-1 bg-gray-200 px-3 py-2 rounded-lg hover:bg-red-500 transition hover:text-white dark:bg-gray-900 dark:hover:bg-red-600">
            <LogOut size={20} />
            <span>Logout</span>
          </button>
        </div>
      </nav>
      {/* Main Content */}
      <div className="container mx-auto py-10 px-4">
        <h1 className="text-3xl font-semibold text-center text-gray-800 dark:text-gray-100">
          Market Sentiment Analysis
        </h1>

        {loading ? (
          <div className="flex justify-center items-center mt-10">
            <Loader2 className="w-10 h-10 animate-spin text-blue-500" />
          </div>
        ) : (
          sentiment && (
            <div className="mt-8 grid md:grid-cols-3 gap-6">
              {/* Positive Sentiment */}
              <Card className="bg-green-200 dark:bg-green-900">
                <CardHeader className="flex flex-row justify-between">
                  <CardTitle className="text-green-700 dark:text-green-300">Positive</CardTitle>
                  <TrendingUp className="text-green-500" />
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{(sentiment.overall_sentiment.financial_sentiment.positive * 100).toFixed(2)}%</p>
                </CardContent>
              </Card>

              {/* Neutral Sentiment */}
              <Card className="bg-gray-200 dark:bg-gray-800">
                <CardHeader className="flex flex-row justify-between">
                  <CardTitle className="text-gray-700 dark:text-gray-300">Neutral</CardTitle>
                  <BarChart className="text-gray-500" />
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{(sentiment.overall_sentiment.financial_sentiment.neutral * 100).toFixed(2)}%</p>
                </CardContent>
              </Card>

              {/* Negative Sentiment */}
              <Card className="bg-red-200 dark:bg-red-900">
                <CardHeader className="flex flex-row justify-between">
                  <CardTitle className="text-red-700 dark:text-red-300">Negative</CardTitle>
                  <TrendingDown className="text-red-500" />
                </CardHeader>
                <CardContent>
                  <p className="text-2xl font-bold">{(sentiment.overall_sentiment.financial_sentiment.negative * 100).toFixed(2)}%</p>
                </CardContent>
              </Card>
            </div>
          )
        )}

        {/* Refresh Button */}
        <div className="mt-6 flex justify-center">
          <Button onClick={()=>{
            fetchSentiment();
            setNewsArticles([]);
          }} variant="outline" className="z-10">
            Refresh Data
          </Button>
        </div>
      </div>
      <div>
      {newsArticles && <h3 className="text-3xl text-center font-semibold">News Analysed</h3>}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4 mt-4 mx-4">
            {newsArticles.map((article, index) => (
              <a
                key={index}
                href={article.article.url}
                target="_blank"
                rel="noopener noreferrer"
                className="block p-3 border rounded-lg shadow-sm hover:shadow-md transition cursor-pointer z-10 bg-white dark:bg-gray-800"
              >
                <h4 className="font-semibold text-green-800 text-2xl dark:text-white">{article.article.title}</h4>
                <p className="text-lg text-gray-700 mt-1 dark:text-gray-400">{article.article.text}</p>
              </a>
            ))}
        </div>
      </div>
    </div>
  );
};

export default RiskAssessment;
