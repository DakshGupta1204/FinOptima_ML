"use client";
import React, { useState } from "react";
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
} from "lucide-react";
import { useRouter } from "next/navigation";

interface Prediction {
  recommended_strategy: string;
  suggested_monthly_payment: number;
  repayment_probability: number;
  estimated_payoff_months: number;
}
const RiskAssessment: React.FC = () => {
  const router = useRouter();
  const [darkMode, setDarkMode] = useState(false); // Theme State
  const [formData, setFormData] = useState({
    income: "",
    debt: "",
    monthly_expenses: "",
    credit_score: "",
  });

  const [prediction, setPrediction] = useState<Prediction | null>(null);

  const handleChange = (e:any) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e:any) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:8000/debt-management", formData);
      console.log(response.data)
      setPrediction(response.data);
    } catch (error) {
      console.error("Error predicting:", error);
    }
  };
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
                                  { name: "Portfolio Optimiser", icon: Calculator, link:"/ml-portfolio" },
                                  { name: "Market Analysis", icon: Bot, link:"/ml-market" },
                                  { name: "Financial Planning", icon: ShieldCheck, link:"/ml-financial" },
                                  { name: "Investment Planner", icon: PiggyBank, link:"/ml-investment" },
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
      <div className="max-w-lg mx-auto mt-10 p-6 bg-white shadow-lg rounded-lg relative z-10 dark:bg-[#161b22] dark:text-white">
      <h2 className="text-2xl font-bold mb-4">Debt Repayment Prediction</h2>
      <form onSubmit={handleSubmit} className="space-y-4 ">
        <input
          type="number"
          name="income"
          placeholder="Income"
          value={formData.income}
          onChange={handleChange}
          required
          className="w-full p-2 border rounded dark:bg-gray-800 dark:text-white"
        />
        <input
          type="number"
          name="debt"
          placeholder="Debt"
          value={formData.debt}
          onChange={handleChange}
          required
          className="w-full p-2 border rounded dark:bg-gray-800 dark:text-white"
        />
        <input
          type="number"
          name="monthly_expenses"
          placeholder="Monthly Expenses"
          value={formData.monthly_expenses}
          onChange={handleChange}
          required
          className="w-full p-2 border rounded dark:bg-gray-800 dark:text-white"
        />
        <input
          type="number"
          name="credit_score"
          placeholder="Credit Score"
          value={formData.credit_score}
          onChange={handleChange}
          required
          className="w-full p-2 border rounded dark:bg-gray-800 dark:text-white"
        />
        <button type="submit" className="w-full bg-primary hover:bg-primary-dark text-white p-2 rounded">
          Predict
        </button>
      </form>
      {prediction !== null && (
        <div className="mt-4 p-3 bg-gray-100 rounded dark:bg-gray-800">
          <h3 className="text-lg font-bold">Prediction Result</h3>
          <p>Recommended Stratergy: <strong>{(prediction.recommended_strategy)}</strong></p>
          <p>Suggested monthly payments: <strong>{(prediction.suggested_monthly_payment).toFixed(2)}</strong></p>
          <p>Estimated months required: <strong>{(prediction.estimated_payoff_months)}</strong></p>
        </div>
      )}
    </div>
    </div>
  );
};

export default RiskAssessment;
