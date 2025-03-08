import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter, ZAxis, Cell } from 'recharts';

// Sample data based on PISA analysis
const countryPerformance = [
  { country: 'Singapore', math: 569, reading: 549, science: 551 },
  { country: 'China', math: 591, reading: 555, science: 590 },
  { country: 'Japan', math: 527, reading: 504, science: 529 },
  { country: 'Korea', math: 526, reading: 514, science: 519 },
  { country: 'Estonia', math: 523, reading: 523, science: 530 },
  { country: 'Finland', math: 507, reading: 520, science: 522 },
  { country: 'Canada', math: 512, reading: 520, science: 518 },
  { country: 'Germany', math: 500, reading: 498, science: 503 },
  { country: 'USA', math: 478, reading: 505, science: 502 },
  { country: 'OECD Average', math: 489, reading: 487, science: 489 }
];

const genderGaps = [
  { subject: 'Mathematics', male: 492, female: 487, gap: 5 },
  { subject: 'Reading', male: 472, female: 502, gap: -30 },
  { subject: 'Science', male: 488, female: 490, gap: -2 }
];

const escsImpact = [
  { country: 'Hungary', correlation: 0.49 },
  { country: 'Luxembourg', correlation: 0.41 },
  { country: 'France', correlation: 0.40 },
  { country: 'Slovakia', correlation: 0.39 },
  { country: 'Germany', correlation: 0.36 },
  { country: 'OECD Average', correlation: 0.32 },
  { country: 'Iceland', correlation: 0.27 },
  { country: 'Canada', correlation: 0.26 },
  { country: 'Estonia', correlation: 0.25 },
  { country: 'Hong Kong', correlation: 0.18 }
];

const scatterData = [
  { x: -1.5, y: 420, z: 10, name: 'Country A' },
  { x: -1.0, y: 440, z: 15, name: 'Country B' },
  { x: -0.5, y: 460, z: 20, name: 'Country C' },
  { x: 0.0, y: 480, z: 25, name: 'Country D' },
  { x: 0.5, y: 500, z: 30, name: 'Country E' },
  { x: 1.0, y: 520, z: 35, name: 'Country F' },
  { x: 1.5, y: 540, z: 40, name: 'Country G' },
];

const PISADashboard = () => {
  const [activeTab, setActiveTab] = useState('countries');
  
  return (
    <div className="flex flex-col w-full min-h-screen bg-gray-50 font-sans">
      <header className="bg-gradient-to-r from-blue-800 to-blue-600 text-white p-6 shadow-lg">
        <div className="max-w-6xl mx-auto">
          <h1 className="text-3xl font-bold tracking-tight">PISA 2018 Data Analysis Dashboard</h1>
          <p className="mt-2 text-blue-100 text-lg">Interactive visualizations of key PISA performance indicators</p>
        </div>
      </header>
      
      <nav className="sticky top-0 z-10 bg-white shadow-md">
        <div className="max-w-6xl mx-auto">
          <div className="flex overflow-x-auto">
            <button 
              className={`py-4 px-6 font-medium text-sm transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50 
                ${activeTab === 'countries' 
                  ? 'text-blue-600 border-b-2 border-blue-600 font-semibold' 
                  : 'text-gray-600 hover:text-blue-500 hover:bg-blue-50'}`}
              onClick={() => setActiveTab('countries')}
            >
              Country Performance
            </button>
            <button 
              className={`py-4 px-6 font-medium text-sm transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50
                ${activeTab === 'gender' 
                  ? 'text-blue-600 border-b-2 border-blue-600 font-semibold' 
                  : 'text-gray-600 hover:text-blue-500 hover:bg-blue-50'}`}
              onClick={() => setActiveTab('gender')}
            >
              Gender Gaps
            </button>
            <button 
              className={`py-4 px-6 font-medium text-sm transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50
                ${activeTab === 'escs' 
                  ? 'text-blue-600 border-b-2 border-blue-600 font-semibold' 
                  : 'text-gray-600 hover:text-blue-500 hover:bg-blue-50'}`}
              onClick={() => setActiveTab('escs')}
            >
              Socioeconomic Impact
            </button>
            <button 
              className={`py-4 px-6 font-medium text-sm transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-50
                ${activeTab === 'relationships' 
                  ? 'text-blue-600 border-b-2 border-blue-600 font-semibold' 
                  : 'text-gray-600 hover:text-blue-500 hover:bg-blue-50'}`}
              onClick={() => setActiveTab('relationships')}
            >
              ESCS-Performance Relationship
            </button>
          </div>
        </div>
      </nav>
      
      <main className="flex-grow p-6 md:p-8">
        <div className="max-w-6xl mx-auto bg-white rounded-lg shadow-md p-6 md:p-8">
          {activeTab === 'countries' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Top Performing Countries across Subjects</h2>
              <p className="mb-6 text-gray-600 leading-relaxed">
                The chart below shows the performance of top countries in PISA 2018 across Mathematics, Reading, and Science. 
                East Asian countries consistently lead in Mathematics and Science, while European countries perform strongly in Reading.
              </p>
              <div className="h-96 mb-8">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={countryPerformance}
                    margin={{ top: 20, right: 30, left: 20, bottom: 70 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="country" angle={-45} textAnchor="end" height={70} tick={{fill: '#666'}} />
                    <YAxis domain={[450, 600]} label={{ value: 'Average Score', angle: -90, position: 'insideLeft', fill: '#666' }} tick={{fill: '#666'}} />
                    <Tooltip cursor={{fill: 'rgba(0, 0, 0, 0.05)'}} />
                    <Legend wrapperStyle={{paddingTop: '20px'}} />
                    <Bar dataKey="math" name="Mathematics" fill="#4F46E5" />
                    <Bar dataKey="reading" name="Reading" fill="#10B981" />
                    <Bar dataKey="science" name="Science" fill="#F59E0B" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 p-6 bg-blue-50 rounded-lg border border-blue-100">
                <h3 className="font-bold text-lg text-gray-800 mb-2">Key Findings:</h3>
                <ul className="space-y-2 mt-3 text-gray-700">
                  <li className="flex items-start">
                    <span className="text-blue-500 mr-2">•</span>
                    <span>East Asian education systems (Singapore, China, Japan, Korea) consistently outperform other regions in mathematics and science</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-500 mr-2">•</span>
                    <span>European countries like Estonia and Finland show strong performance across all domains</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-blue-500 mr-2">•</span>
                    <span>The OECD average falls significantly below top performers, highlighting global achievement gaps</span>
                  </li>
                </ul>
              </div>
            </div>
          )}
          
          {activeTab === 'gender' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Gender Performance Gaps Across Subjects</h2>
              <p className="mb-6 text-gray-600 leading-relaxed">
                This chart visualizes the average performance of male and female students across the three core PISA domains.
                A significant reading advantage for girls is observed globally, while mathematics shows a slight advantage for boys.
              </p>
              <div className="h-96 mb-8">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={genderGaps}
                    margin={{ top: 20, right: 30, left: 20, bottom: 30 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="subject" tick={{fill: '#666'}} />
                    <YAxis domain={[460, 510]} label={{ value: 'Average Score', angle: -90, position: 'insideLeft', fill: '#666' }} tick={{fill: '#666'}} />
                    <Tooltip cursor={{fill: 'rgba(0, 0, 0, 0.05)'}} />
                    <Legend wrapperStyle={{paddingTop: '20px'}} />
                    <Bar dataKey="male" name="Boys" fill="#3B82F6" />
                    <Bar dataKey="female" name="Girls" fill="#EC4899" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 p-6 bg-pink-50 rounded-lg border border-pink-100">
                <h3 className="font-bold text-lg text-gray-800 mb-2">Key Findings:</h3>
                <ul className="space-y-2 mt-3 text-gray-700">
                  <li className="flex items-start">
                    <span className="text-pink-500 mr-2">•</span>
                    <span>Girls outperform boys in reading by approximately 30 points on average, a significant gap equivalent to about a year of schooling</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-pink-500 mr-2">•</span>
                    <span>Boys maintain a small advantage in mathematics (5 points on average)</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-pink-500 mr-2">•</span>
                    <span>Science performance is nearly equal between genders, with girls holding a slight 2-point advantage</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-pink-500 mr-2">•</span>
                    <span>The reading gender gap is consistent across nearly all participating countries</span>
                  </li>
                </ul>
              </div>
            </div>
          )}
          
          {activeTab === 'escs' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Impact of Socioeconomic Status on Performance by Country</h2>
              <p className="mb-6 text-gray-600 leading-relaxed">
                This chart shows the correlation between socioeconomic status (ESCS index) and mathematics performance across countries.
                Higher values indicate a stronger relationship between a student's socioeconomic background and their academic performance.
              </p>
              <div className="h-96 mb-8">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart
                    data={escsImpact}
                    layout="vertical"
                    margin={{ top: 20, right: 30, left: 100, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis type="number" domain={[0, 0.5]} tick={{fill: '#666'}} />
                    <YAxis type="category" dataKey="country" tick={{fill: '#666'}} />
                    <Tooltip cursor={{fill: 'rgba(0, 0, 0, 0.05)'}} />
                    <Legend wrapperStyle={{paddingTop: '20px'}} />
                    <Bar dataKey="correlation" name="ESCS-Math Correlation">
                      {escsImpact.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.correlation > 0.32 ? '#EF4444' : '#10B981'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 p-6 bg-green-50 rounded-lg border border-green-100">
                <h3 className="font-bold text-lg text-gray-800 mb-2">Key Findings:</h3>
                <ul className="space-y-2 mt-3 text-gray-700">
                  <li className="flex items-start">
                    <span className="text-green-500 mr-2">•</span>
                    <span>Countries with higher correlations (like Hungary and Luxembourg) show greater educational inequality, where socioeconomic background strongly predicts academic success</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 mr-2">•</span>
                    <span>Countries with lower correlations (like Hong Kong and Estonia) demonstrate more equitable education systems, where students can succeed regardless of socioeconomic background</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-green-500 mr-2">•</span>
                    <span>The OECD average correlation of 0.32 indicates that approximately 10% of variation in student performance can be explained by socioeconomic factors</span>
                  </li>
                </ul>
              </div>
            </div>
          )}
          
          {activeTab === 'relationships' && (
            <div>
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Relationship Between Socioeconomic Status and Mathematics Performance</h2>
              <p className="mb-6 text-gray-600 leading-relaxed">
                This scatter plot visualizes the relationship between a country's average socioeconomic status (ESCS) and their mathematics performance.
                The size of each point represents the performance gap between high and low ESCS students within that country.
              </p>
              <div className="h-96 mb-8">
                <ResponsiveContainer width="100%" height="100%">
                  <ScatterChart
                    margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis 
                      type="number" 
                      dataKey="x" 
                      name="ESCS Index" 
                      domain={[-2, 2]}
                      label={{ value: 'Country Average ESCS Index', position: 'bottom', offset: 0, fill: '#666' }}
                      tick={{fill: '#666'}}
                    />
                    <YAxis 
                      type="number" 
                      dataKey="y" 
                      name="Math Score" 
                      domain={[400, 550]}
                      label={{ value: 'Mathematics Performance', angle: -90, position: 'insideLeft', fill: '#666' }}
                      tick={{fill: '#666'}}
                    />
                    <ZAxis 
                      type="number" 
                      dataKey="z" 
                      range={[60, 400]} 
                      name="Performance Gap"
                    />
                    <Tooltip 
                      cursor={{ strokeDasharray: '3 3' }}
                      formatter={(value, name) => {
                        if (name === 'Math Score') return [value, 'Mathematics Score'];
                        if (name === 'ESCS Index') return [value, 'Average ESCS'];
                        if (name === 'Performance Gap') return [value, 'SES Performance Gap'];
                        return [value, name];
                      }}
                      labelFormatter={(label) => scatterData[label].name}
                    />
                    <Legend wrapperStyle={{paddingTop: '20px'}} />
                    <Scatter name="Countries" data={scatterData} fill="#6366F1" />
                  </ScatterChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 p-6 bg-indigo-50 rounded-lg border border-indigo-100">
                <h3 className="font-bold text-lg text-gray-800 mb-2">Key Findings:</h3>
                <ul className="space-y-2 mt-3 text-gray-700">
                  <li className="flex items-start">
                    <span className="text-indigo-500 mr-2">•</span>
                    <span>There is a positive correlation between country-level socioeconomic status and average mathematics performance</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-indigo-500 mr-2">•</span>
                    <span>Countries with similar ESCS levels can have significantly different performance outcomes, indicating that policy and education system characteristics matter</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-indigo-500 mr-2">•</span>
                    <span>The size of within-country performance gaps (indicated by circle size) varies considerably, with some countries showing much larger inequalities than others</span>
                  </li>
                  <li className="flex items-start">
                    <span className="text-indigo-500 mr-2">•</span>
                    <span>Some countries manage to achieve both high performance and relatively small socioeconomic gaps</span>
                  </li>
                </ul>
              </div>
            </div>
          )}
        </div>
      </main>
      
      <footer className="bg-gray-800 text-white py-8 mt-auto">
        <div className="max-w-6xl mx-auto px-6 md:px-8">
          <div className="grid md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold mb-3">PISA 2018 Data Analysis Project</h3>
              <p className="text-gray-300">Created with Recharts and React</p>
            </div>
            <div>
              <h3 className="text-lg font-semibold mb-3">Data Sources</h3>
              <p className="text-gray-300">OECD Programme for International Student Assessment (PISA) 2018</p>
              <p className="text-gray-400 text-sm mt-2">All data and visualizations are based on official PISA 2018 results</p>
            </div>
          </div>
          <div className="border-t border-gray-700 mt-6 pt-6 text-sm text-gray-400">
            <p>© 2025 PISA Dashboard | Educational use permitted</p>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default PISADashboard;