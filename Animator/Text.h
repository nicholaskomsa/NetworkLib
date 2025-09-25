#pragma once

#include <GL/glew.h>
#include <freetype/freetype.h>

#include <string>
#include <vector>
#include <ranges>
#include <algorithm>
#include <numeric>

#include "Geometry.h"

class Text {
public:

	std::size_t mX = 0, mY = 0;

	FT_Int mTotalPixelsWidth = 0, mTotalPixelsHeight = 0, mMaxAscent = 0, mMaxDescent = 0;
	std::vector<std::uint32_t> mTextPixelsBuffer;

	bool create( const std::string& text, const FT_Face& face) {

		FT_Int oldWidth = mTotalPixelsWidth, oldHeight = mTotalPixelsHeight;

		mTotalPixelsWidth = 0;
		mMaxAscent = 0;
		mMaxDescent = 0;
		
		for (auto c: text ){

			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;

			mTotalPixelsWidth += g->advance.x >> 6; // advance in pixels
			mMaxAscent = std::max(mMaxAscent, g->bitmap_top);
			mMaxDescent = std::max(mMaxDescent, FT_Int(g->bitmap.rows - g->bitmap_top));
		}
		mTotalPixelsHeight = mMaxAscent + mMaxDescent;

		std::uint32_t greyColor = 0xFFFFFFFF;//opaque with white background with grey forground/text
		std::uint8_t* bytes = reinterpret_cast<std::uint8_t*>(&greyColor);

		//add spaces between glyphs and padding of line and bottom of line
		mTotalPixelsWidth += text.size()*2 +2 ;
		mTotalPixelsHeight += 2;
		mTextPixelsBuffer.clear();
		mTextPixelsBuffer.resize(mTotalPixelsWidth * mTotalPixelsHeight, greyColor);

		int penX = 0;
		for (std::size_t i : std::views::iota(0ULL, text.size())) {

			char c = text[i];

			FT_Load_Char(face, c, FT_LOAD_RENDER);
			FT_GlyphSlot g = face->glyph;
			FT_Bitmap& bmp = g->bitmap;

			int xOffset = penX + g->bitmap_left;
			int yOffset = mMaxAscent - g->bitmap_top;

			for (int row = 0; row < bmp.rows; ++row) {
				for (int col = 0; col < bmp.width; ++col) {
					int x = 1 + xOffset + col + i*2; //+i/1 == space between glyphs
					int y = 1 + yOffset + row;

					float greyScale = 1.0f - bmp.buffer[row * bmp.pitch + col] / 255.0f;
					std::uint8_t* savedColor = reinterpret_cast<std::uint8_t*>(&mTextPixelsBuffer[y * mTotalPixelsWidth + x]);

					savedColor[0] = bytes[0] * greyScale; // R
					savedColor[1] = bytes[1] * greyScale;   // G
					savedColor[2] = bytes[2] * greyScale; // B
					savedColor[3] = bytes[3]; // A
				}
			}

			penX += g->advance.x >> 6;
		}

		//return if size increased
		return oldWidth < mTotalPixelsWidth || oldHeight < mTotalPixelsHeight;
	} 
	void destroy() {
		mTextPixelsBuffer.clear();
		mTextPixelsBuffer.shrink_to_fit();
	}
	float getAspectRatio() {
		return mTotalPixelsWidth / float(mTotalPixelsHeight);
	}
};

class TextArea {
public:
	std::vector<Text> mLabels;
	using LabelReference = std::size_t;

	using LabeledValue = std::pair<Text, Text>;
	std::vector<LabeledValue> mLabeledValues;
	using LabeledValueReference = std::size_t;

	QuadManager::QuadReference mQuadReference;

	GLuint mTexture = 0;
	FT_Int mTotalPixelsWidth = 0, mTotalPixelsHeight = 0;

	std::size_t mInsertY = 0; //start at bottom

	float mScale = 0.01f;

	void create(QuadManager& qm, float scale = 1.0) {
		mScale = scale;

		calculateDimensions();

		auto YLineNum = mLabeledValues.size() + mLabels.size();
		mQuadReference = qm.add(-1.0, -1.0, mTotalPixelsWidth / float(mTotalPixelsHeight), scale * YLineNum);
	}

	void createTexture(std::size_t width, std::size_t height) {

		if (mTexture)
			glDeleteTextures(1, &mTexture);

		glGenTextures(1, &mTexture);

		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, mTexture);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, mTotalPixelsWidth, mTotalPixelsHeight, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, nullptr);

		GLubyte color[4] = { 255,255,255,255 };
		glClearTexImage(mTexture, 0, GL_RGBA, GL_UNSIGNED_BYTE, color);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glBindTexture(GL_TEXTURE_2D, 0);
	}
	void destroy() {

		for (auto& label : mLabels)
			label.destroy();

		for (auto& [caption, value] : mLabeledValues) {

			caption.destroy();
			value.destroy();
		}

		if (mTexture) {
			glDeleteTextures(1, &mTexture);
			mTexture = 0;
		}
	}

	void unionDimensions(const Text& text) {
		mTotalPixelsWidth = std::max(mTotalPixelsWidth, FT_Int( text.mX + text.mTotalPixelsWidth));
		mTotalPixelsHeight = std::max(mTotalPixelsHeight, FT_Int( text.mY + text.mTotalPixelsHeight));
	}
	void calculateDimensions() {

		//loop through all text components and get the bounding rectangle including the old one, get bigger not smaller

		FT_Int oldWidth = mTotalPixelsWidth, oldHeight = mTotalPixelsHeight;

		for (auto& label : mLabels)
			unionDimensions(label);

		for (auto& [label, caption] : mLabeledValues) {
			unionDimensions(label);
			unionDimensions(caption);
		}

		if (mTotalPixelsWidth == oldWidth && mTotalPixelsHeight == oldHeight) return;

		createTexture(mTotalPixelsWidth, mTotalPixelsHeight);
	}
	void render(QuadManager& qm, bool updateLabels = true) {

		glBindTexture(GL_TEXTURE_2D, mTexture);

		auto updateTexture = [&](auto& text) {
			glTexSubImage2D(GL_TEXTURE_2D, 0, text.mX, text.mY, text.mTotalPixelsWidth, text.mTotalPixelsHeight, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, text.mTextPixelsBuffer.data());
			};
		
		if (updateLabels) {
			for (auto& label : mLabels)
				updateTexture(label);

			for (auto& [label, value] : mLabeledValues)
				updateTexture(label);
		}
		for (auto& [label, value] : mLabeledValues)
			updateTexture(value);

		qm.render(mQuadReference);
	} 

	LabelReference addLabel(FT_Face& fontFace, const std::string& text) {
		Text t;
		t.create(text, fontFace);
		t.mX = 0;
		t.mY = mInsertY;
		mInsertY += t.mTotalPixelsHeight;

		mLabels.push_back(std::move(t));

		return mLabels.size() - 1;
	}

	LabeledValueReference addLabeledValue(FT_Face& fontFace, const std::string& text, const std::string& valueText) {
		
		std::size_t insertX = 0;

		Text caption;
		caption.create(text, fontFace);
		caption.mX = 0; 
		caption.mY = mInsertY;

		insertX = caption.mTotalPixelsWidth;

		Text value;
		value.create(valueText, fontFace);
		value.mY = mInsertY;
		value.mX = insertX;

		mLabeledValues.push_back({ std::move(caption), std::move(value) });
		                 
		mInsertY += value.mTotalPixelsHeight;

		return mLabeledValues.size() - 1;
	}

	void updateLabeledValue(LabeledValueReference labeledValueRef, const std::string& valueText, const FT_Face& fontFace) {
		auto& value = mLabeledValues[labeledValueRef].second;
		auto resized = value.create(valueText, fontFace);
		if (resized) calculateDimensions();

	}
	void updateLabel(LabelReference labelRef, const std::string& labelText, const FT_Face& fontFace) {
		auto& label = mLabels[labelRef];
		auto resized = label.create(labelText, fontFace);
		if (resized) calculateDimensions();
	}
};
class TextManager {

public:

	QuadManager* mQuadManager;

	FT_Library mFreeType;
	FT_Face mFontFace;

	using TextAreaReference = std::size_t;
	std::vector<TextArea> mTextAreas;

	float mInsertY = -1.0f; //start at bottom
	bool mVisible = true;

	void create(QuadManager* quadManager, const std::string& fontName, FT_UInt fontSize, float scale = 0.01f) {

		mQuadManager = quadManager;
		
		FT_Init_FreeType(&mFreeType);

		FT_New_Face(mFreeType, fontName.c_str(), 0, &mFontFace);
		FT_Set_Pixel_Sizes(mFontFace, 0, fontSize);
	}

	void destroy() {

		FT_Done_Face(mFontFace);       // Destroys the font face
		FT_Done_FreeType(mFreeType);     // Shuts down the FreeType library
	}

	TextAreaReference addTextArea() {
		mTextAreas.push_back({});
		return mTextAreas.size()-1;
	}
	TextArea& getTextArea(TextAreaReference ref) {
		return mTextAreas[ref];
	}
	void render() {

		if (!mVisible) return;

		auto& qm = *mQuadManager;
		 
		//glEnable(GL_BLEND);
		//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		for (auto& txtArea : mTextAreas)
			txtArea.render(qm);

		glBindTexture(GL_TEXTURE_2D, 0);
	} 
	void show() {
		mVisible = true;
	}
	void hide() {
		mVisible = false;
	}
	void toggleVisibility() {
		mVisible = !mVisible;
	}
};