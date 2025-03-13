"use client";
import { useState, useEffect } from "react";
import { 
    Sun, Moon, User, Lock, Mail, Save, Edit2, X, Wallet, PiggyBank, LineChart, Target, Brain, Shield,ChartBar
} from "lucide-react";
import { useRouter } from "next/navigation";

const UserProfile = () => {
    const router = useRouter();
    const [darkMode, setDarkMode] = useState(false);
    const [isEditing, setIsEditing] = useState(false);
    const [showModal, setShowModal] = useState(false);
    const [userData, setUserData] = useState({
        name: "John Doe",
        email: "johndoe@example.com",
        role: "Investor",
        bio: "Passionate about financial markets and data-driven investment strategies.",
        password: "********",
    });

    useEffect(() => {
        if (darkMode) {
            document.documentElement.classList.add("dark");
        } else {
            document.documentElement.classList.remove("dark");
        }
    }, [darkMode]);

    const toggleDarkMode = () => setDarkMode(!darkMode);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        setUserData({ ...userData, [e.target.name]: e.target.value });
    };

    const handleSave = () => {
        setIsEditing(false);
        console.log("Updated User Data:", userData);
    };
    const backgroundIcons = [
        { Icon: Mail, size: 200, top: 60, left: 40, rotate: "rotate-12" },
        { Icon: Lock, size: 150, bottom: 40, right: 60, rotate: "-rotate-12" },
        { Icon: ChartBar, size: 180, top: "7%", left: "50%", rotate: "rotate-135" },
        { Icon: Wallet, size: 160, top: "30%", right: "1%", rotate: "-rotate-15" },
        { Icon: PiggyBank, size: 140, bottom: "1%", left: "20%", rotate: "rotate-30" },
        { Icon: LineChart, size: 170, top: "2%", right: "18%", rotate: "-rotate-20" },
        { Icon: Target, size: 130, bottom: "30%", right: "25%", rotate: "rotate-25" },
        { Icon: Brain, size: 150, top: "60%", left: "1%", rotate: "-rotate-35" },
        { Icon: Shield, size: 120, bottom: "5%", left: "55%", rotate: "rotate-15" },
    ];
    return (
        <div className="w-full min-h-screen bg-gray-100 dark:bg-[#0d1117] transition-all duration-300 flex flex-col">
            {/* Navbar */}
            <nav className="flex justify-between items-center p-8 pb-0 z-10">
                <div className="flex items-center space-x-2 cursor-pointer" onClick={()=>router.push("/")}>
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
                </div>
            </nav>
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
            {/* Main Content */}
            <div className={`flex flex-col lg:flex-row justify-center items-start lg:items-center mt-12 px-8 gap-8 z-10 ${showModal ? "filter blur-sm" : ""}`}>
                {/* User Profile Section */}
                <div className="bg-white dark:bg-[#161b22] shadow-xl rounded-2xl p-10 max-w-xl w-full">
                    <div className="flex flex-col items-center space-y-4">
                        {/* Avatar */}
                        <div className="w-24 h-24 rounded-full bg-primary-light flex items-center justify-center text-3xl font-bold text-gray-600">
                            {userData.name[0]}
                        </div>

                        {/* User Info */}
                        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">{userData.name}</h1>
                        <p className="text-primary dark:text-gray-300">{userData.role}</p>
                    </div>

                    {/* Editable Fields */}
                    <div className="mt-6 space-y-4">
                        {/* Name */}
                        <div className="flex items-center border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2 bg-gray-100 dark:bg-gray-800">
                            <User className="w-5 h-5 text-primary dark:text-gray-400" />
                            <input
                                type="text"
                                name="name"
                                value={userData.name}
                                onChange={handleChange}
                                className="bg-transparent border-none outline-none ml-2 w-full text-gray-900 dark:text-white"
                                disabled={!isEditing}
                            />
                        </div>

                        {/* Email */}
                        <div className="flex items-center border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2 bg-gray-100 dark:bg-gray-800">
                            <Mail className="w-5 h-5 text-primary dark:text-gray-400" />
                            <input
                                type="email"
                                name="email"
                                value={userData.email}
                                onChange={handleChange}
                                className="bg-transparent border-none outline-none ml-2 w-full text-gray-900 dark:text-white"
                                disabled
                            />
                        </div>

                        {/* Bio */}
                        <textarea
                            name="bio"
                            value={userData.bio}
                            onChange={handleChange}
                            className="w-full border border-gray-300 dark:border-gray-700 rounded-lg px-4 py-2 bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white outline-none"
                            rows={3}
                            disabled={!isEditing}
                        />
                    </div>

                    {/* Buttons */}
                    <div className="flex justify-between mt-6">
                        {isEditing ? (
                            <button
                                onClick={handleSave}
                                className="bg-primary hover:bg-primary-dark text-white px-6 py-2 rounded-lg transition-colors flex items-center gap-2"
                            >
                                <Save className="w-5 h-5" /> Save Changes
                            </button>
                        ) : (
                            <button
                                onClick={() => setIsEditing(true)}
                                className="bg-primary hover:bg-primary-dark text-white px-6 py-2 rounded-lg transition-colors flex items-center gap-2"
                            >
                                <Edit2 className="w-5 h-5" /> Edit Profile
                            </button>
                        )}
                    </div>
                </div>

                {/* Financial Options Section */}
                <div className="bg-white dark:bg-[#161b22] shadow-xl rounded-2xl p-10 max-w-xl w-full">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">Financial Details</h2>
                    <div className="space-y-2">
                        <p className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
                            <Shield className="w-5 h-5 text-primary dark:text-gray-400" /> Risk Tolerance: Medium
                        </p>
                        <p className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
                            <Brain className="w-5 h-5 text-primary dark:text-gray-400" /> Market Experience: 5 Years
                        </p>
                        <p className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
                            <Wallet className="w-5 h-5 text-primary dark:text-gray-400" /> Funds Value: $150,000
                        </p>
                        <p className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
                            <PiggyBank className="w-5 h-5 text-primary dark:text-gray-400" /> Debt: $30,000
                        </p>
                        <p className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
                            <Target className="w-5 h-5 text-primary dark:text-gray-400" /> Age: 35
                        </p>
                    </div>

                    {/* Link to Market Cluster Analysis */}
                    <button
                        onClick={() => setShowModal(true)}
                        className="mt-6 text-primary dark:text-primary-light hover:underline"
                    >
                        Market Cluster Analysis & Financial Literacy Path
                    </button>
                </div>
            </div>

            {/* Modal for ML Tool */}
            {showModal && (
                <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-20">
                    <div className="bg-white dark:bg-[#161b22] p-6 rounded-lg max-w-md w-full shadow-lg">
                        <div className="flex justify-between items-center">
                            <h3 className="text-xl font-bold text-gray-900 dark:text-white">Market Analysis Tool</h3>
                            <X className="cursor-pointer text-gray-500 dark:text-gray-400" onClick={() => setShowModal(false)} />
                        </div>
                        <p className="mt-4 text-gray-700 dark:text-gray-300">
                        This <b>Investment Learning System</b> processes user financial data, clusters users based on investment preferences, runs **Monte Carlo simulations** to predict portfolio outcomes, and generates personalized **learning paths**. It uses **machine learning** (Agglomerative Clustering, PCA) for user segmentation, **statistical simulations** to model investment returns with risk-adjusted parameters, and **data visualizations** via Plotly. The system dynamically customizes financial education based on **risk tolerance, income, and investment goals**, helping users make informed decisions. It also logs activities, handles configuration files, and ensures smooth processing of missing or erroneous data.
                        </p>
                        <a href="/ml-tool" className="mt-4 block text-blue-600 dark:text-blue-400 underline">Go to Tool</a>
                    </div>
                </div>
            )}
        </div>
    );
};

export default UserProfile;
